import numpy as np
from pydispatch import dispatcher
from typing import Dict, List

from agent_components.demand.estimator import CustomerPredictions
from agent_components.wholesale.environments.PowerTacEnv import log, PowerTacEnv
from agent_components.wholesale.environments.PowerTacWholesaleAgent import PowerTacWholesaleAgent
from agent_components.wholesale.util import calculate_running_average
from communication.grpc_messages_pb2 import PBMarketTransaction, PBTimeslotUpdate, PBClearedTrade, \
    PBMarketBootstrapData, PBTariffTransaction, PBTxType
from communication.pubsub import signals as signals
from communication.pubsub.SignalConsumer import SignalConsumer
from util import config as cfg
from util.learning_utils import tensorboard_write_mpd_sum


class WholesaleEnvironmentManager(SignalConsumer):
    """This class ties together the powertac side of RL and the "classic" side of RL, i.e. the agent driven side.
    Agents in the classic literature step the environment after they have made a decision. PowerTAC doesn't wait for an agent.
    If the agent is too slow, shit goes on. So this ties it all together. """

    def __init__(self, agent: PowerTacWholesaleAgent, reward_function):
        super().__init__()
        self.environments: Dict[int, "PowerTacEnv"] = {}  # a map of Environments. Key is the target timestep
        self.reward_function = reward_function
        self.agent: PowerTacWholesaleAgent = agent

        self.historical_average_prices = {}  # a map of arrays

    def subscribe(self):
        """Subscribe to any newly incoming messages from the server"""
        dispatcher.connect(self.handle_market_transaction, signal=signals.PB_MARKET_TRANSACTION)
        dispatcher.connect(self.handle_timeslot_update, signal=signals.PB_TIMESLOT_UPDATE)
        dispatcher.connect(self.handle_cleared_trade, signal=signals.PB_CLEARED_TRADE)
        dispatcher.connect(self.handle_predictions, signal=signals.COMP_USAGE_EST)
        dispatcher.connect(self.handle_tariff_transaction, signal=signals.PB_TARIFF_TRANSACTION)
        dispatcher.connect(self.handle_market_bootstrap_data, signal=signals.PB_MARKET_BOOTSTRAP_DATA)
        log.info("env manager is listenening")
        # TODO
        # tells imbalance after DU has performed balancing
        # dispatcher.connect(None,signal=signals.PB_BALANCE_REPORT)
        # tells the component that the next iteration needs to be calculated.
        # dispatcher.connect(None,signal=signals.PB_TIMESLOT_COMPLETE)

    def unsubscribe(self):
        """unsubscribe from all pubsub messages"""
        dispatcher.disconnect(self.handle_market_transaction, signal=signals.PB_MARKET_TRANSACTION)
        dispatcher.disconnect(self.handle_timeslot_update, signal=signals.PB_TIMESLOT_UPDATE)
        dispatcher.disconnect(self.handle_cleared_trade, signal=signals.PB_CLEARED_TRADE)
        dispatcher.disconnect(self.handle_market_bootstrap_data, signal=signals.PB_MARKET_BOOTSTRAP_DATA)
        dispatcher.disconnect(self.handle_tariff_transaction, signal=signals.PB_TARIFF_TRANSACTION)

    def handle_market_transaction(self, sender, signal: str, msg: PBMarketTransaction):
        if msg.timeslot in self.environments:
            self.environments[msg.timeslot].handle_market_transaction(msg)
        else:
            # this should not happen
            raise Exception("missing environment detected")

    def handle_timeslot_update(self, sender, signal: str, msg: PBTimeslotUpdate):
        """Comes in at every new timestep. Marks a new tick. If no MarketTransactions have come in until now,
        we won't see any for that TS --> no trade"""
        # learn on all "previously active" environments
        for env in self.environments.values():
            env.handle_timeslot_update(msg)
        # then remove just finished ts environment because it's not active anymore
        for ts in list(self.environments.keys()):
            if ts not in range(msg.firstEnabled, msg.lastEnabled + 1):
                #write some summaries before deleting
                tensorboard_write_mpd_sum(self.environments[ts])
                del self.environments[ts]

        # create new environments for all newly active
        # pass agent to each environment, they need the agent to interact with it.
        for ts in range(msg.firstEnabled, msg.lastEnabled + 1):
            if ts not in self.environments:
                self.environments[ts] = PowerTacEnv(self.agent, self.reward_function, ts, self.get_historical_prices(ts))

    def handle_cleared_trade(self, sender, signal: str, msg: PBClearedTrade):
        if msg.timeslot in self.environments:
            self.append_historical(msg)
            self.environments[msg.timeslot].handle_cleared_trade(msg)
        else:
            # this arrives once after timeslot_update without any meaning. so a missing env here is no prob.
            pass

    def handle_predictions(self, sender, signal: str, msg: List[CustomerPredictions]):
        """msg is an array of predictions for the next 24 ts for a customer. Don't care which customer, just care
        how much we need to buy here."""

        # assumes next x timesteps
        preds = self.get_sums_from_preds(msg)
        for ts in preds:
            self.environments[ts].handle_prediction(preds[ts])

    def handle_market_bootstrap_data(self, sender, signal: str, msg: PBMarketBootstrapData):
        """bootstrap data needed to be successful in the first few steps already"""
        log.info("handling market bootstrap data. It's getting serious!")
        c = len(msg.marketPrice)
        assert c == len(msg.mwh)
        for i in range(c):
            pair = [msg.mwh[i], msg.marketPrice[i]]
            self.historical_average_prices[i] = [pair]

    def handle_tariff_transaction(self, sender, signal: str, msg: PBTariffTransaction):
        if (msg.txType is PBTxType.Value("CONSUME") or msg.txType is PBTxType.Value("PRODUCE")) and msg.postedTimeslot in self.environments:
            self.environments[msg.postedTimeslot].handle_tariff_transaction(msg)


    def get_avg_for_ts(self, ts):
        """gets the average price for the given ts based on the historical clearedTrade data"""
        ts_data = np.array(self.historical_average_prices[ts])
        return calculate_running_average(ts_data)

    def append_historical(self, msg: PBClearedTrade):
        """adds a cleared trade to the historical stats data"""
        ts = msg.timeslot
        if ts not in self.historical_average_prices:
            self.historical_average_prices[ts] = []
        self.historical_average_prices[ts].append([msg.executionMWh, msg.executionPrice])

    def get_historical_prices(self, target_ts):
        start = target_ts - cfg.WHOLESALE_HISTORICAL_DATA_LENGTH
        # minimum timestep is one
        start = start if start >= 1 else 1
        self._ensure_historicals_present(target_ts)

        # in the bootstrap situation, we have 336 timeslots and for some reason start at 360. Therefore, it's 24h "lost"
        # TODO is it really?

        avgs = [calculate_running_average(np.array(self.historical_average_prices[i])) for i in range(start, target_ts)]
        # case A: no data --> just zeros
        if len(avgs) == 0:
            return np.zeros(cfg.WHOLESALE_HISTORICAL_DATA_LENGTH)
        # padding at beginning with the first value of avgs
        # case B: some padding needed
        whole = np.zeros(cfg.WHOLESALE_HISTORICAL_DATA_LENGTH)
        whole.fill(avgs[0])
        whole[-len(avgs):] = avgs
        # "case" C: no padding needed
        return whole

    def get_sums_from_preds(self, preds_list: List[CustomerPredictions]) -> Dict[int, float]:
        """
        Calculates the sum per ts from the list of customerpredictions
        :param preds_list:
        :return:
        """
        # iterate over customers
        ts_preds = {}
        # iterate over customers
        for cust in preds_list:
            cu_preds = cust.predictions
            # iterate over ts of each customer
            for ts in cu_preds:
                if ts not in ts_preds:
                    ts_preds[ts] = 0
                ts_preds[ts] += cu_preds[ts]
        return ts_preds

    def _ensure_historicals_present(self, target_ts):
        for ts in range(target_ts - cfg.WHOLESALE_HISTORICAL_DATA_LENGTH, target_ts):
            if ts not in self.historical_average_prices:
                self.historical_average_prices[ts] = []