from collections import deque

import logging
import numpy as np
from gym import Env, spaces
from gym.spaces import Box
from pydispatch import dispatcher
from typing import Dict, Generator, List, Tuple

import communication.pubsub.signals as signals
import util.config as cfg
from agent_components.demand.estimator import CustomerPredictions
from agent_components.wholesale.util import calculate_running_average
from communication.grpc_messages_pb2 import PBClearedTrade, PBMarketBootstrapData, PBMarketTransaction, PBOrder, \
    PBOrderbook, PBTimeslotUpdate, PBTariffTransaction, PBTxType
from communication.pubsub.PubSubTypes import SignalConsumer

log = logging.getLogger(__name__)


class PowerTacWholesaleAgent:
    """Abstract wholesale agent that can act in a `PowerTacEnv`"""

    def forward(self, env: "PowerTacEnv"):
        """Gets an action based on the environment. The agent is responsible for interpreting the environment"""
        raise NotImplementedError

    def backward(self, env: "PowerTacEnv", action, reward):
        """Receive the reward for the action and learn from historical data"""
        raise NotImplementedError

    def learn(self):
        #importing locally to avoid circular dependency
        from agent_components.wholesale.environments.LogEnvManagerAdapter import LogEnvManagerAdapter
        """Gets called by the `main.py` script. Here, the agent creates a new adapter for the logs and passes itself in."""
        adapter = LogEnvManagerAdapter(self)
        # starts to listen to PBOrder events
        adapter.subscribe()
        # and start the loop
        adapter.start()

        # when we arrive here, we're done playing games (no pun intended) and the agent should save it's state
        self.save_model()

    def save_model(self):
        raise NotImplementedError


class WholesaleEnvironmentManager(SignalConsumer):
    """This class ties together the powertac side of RL and the "classic" side of RL, i.e. the agent driven side.
    Agents in the classic literature step the environment after they have made a decision. PowerTAC doesn't wait for an agent.
    If the agent is too slow, shit goes on. So this ties it all together. """

    def __init__(self, agent: PowerTacWholesaleAgent):
        super().__init__()
        self.environments: Dict[int, "PowerTacEnv"] = {}  # a map of Environments. Key is the target timestep
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
                del self.environments[ts]

        # create new environments for all newly active
        # pass agent to each environment, they need the agent to interact with it.
        for ts in range(msg.firstEnabled, msg.lastEnabled + 1):
            if ts not in self.environments:
                self.environments[ts] = PowerTacEnv(self.agent, ts, self.get_historical_prices(ts))

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


class PowerTacEnv(Env):
    """A close resembling of the OpenAI Gym but in PowerTAC, the flow of execution is reversed. This means the agent doesn't call the environment but the other way around
    """

    def __init__(self, agent: PowerTacWholesaleAgent, target_ts, historical_prices):
        super().__init__()
        # final
        self.internals = []
        self.agent = agent
        self.realized_usage = 0
        self._target_timeslot = target_ts
        # changing
        self._historical_prices = deque(maxlen=cfg.WHOLESALE_HISTORICAL_DATA_LENGTH)
        self._historical_prices.extend(historical_prices)

        self._step = 1
        self.orderbooks: List[PBOrderbook] = []  #
        self.purchases: List[PBMarketTransaction] = []
        self.cleared_trades: List[PBClearedTrade] = []
        self.observations: List[PowerTacWholesaleObservation] = []
        #the functional (powertac logic) actions, i.e. mWh, price
        self.actions: List[np.array] = []
        #the raw action values returned by the NN
        self.nn_actions = []
        self.predictions: List[float] = []

    def step(self, action) -> Generator:
        # TODO crit > implement and test
        raise NotImplementedError

    def reset(self):
        """Doesn't do anything, our envs never get reset, just run once and then dismissed"""
        pass

    def render(self, mode='logging'):
        """nothing to render in powertac"""
        pass

    def close(self):
        raise NotImplementedError

    def handle_timeslot_update(self, msg: PBTimeslotUpdate):
        last_action = self.actions[-1] if self.actions else None
        last_observation = self.observations[-1] if self.observations else None
        reward = self.calculate_reward()
        # TODO crit > calculate reward!
        if last_observation is None or last_action is None or self.realized_usage == 0:
            return
        self.agent.backward(self, last_observation, last_action, reward)
        self._step += 1

    def calculate_reward(self):
        from agent_components.wholesale.learning.reward_functions import market_relative_prices, step_close_to_prediction_reward
        if self._step == 25:  # terminal step
            # calculate terminal reward
            rew = market_relative_prices(self)

            log.info("agent reward for terimal state {}".format(rew))
            return rew
        else:
            return step_close_to_prediction_reward(self)
            # calculate intermediate reward


    def handle_prediction(self, prediction):
        """
        at this point the broker is ready to calculate an action! This component is not aware of the exact interests
        of the broker in the environment and therefore it's best to hand the broker everything we know about the environment.
        That usually includes:
        - historical price data of the last X time steps
        - historical price data of the last Y market closings for the target timestep
        - historical predictions for the target timestep
        - historical Orderbooks for the target timestep
        - historical purchases for the target timeslot

        :param prediction: The newest prediction for this target step
        :return:
        """
        log.debug("predictions received in wholesale, starting trading actions")
        self.predictions.append(prediction)
        action, nn_action, internals = self.agent.forward(self)
        self.internals.append(internals)
        # store the action and observation
        self.actions.append(action)
        self.nn_actions.append(nn_action)
        self.send_order(action)

    def handle_cleared_trade(self, msg: PBClearedTrade):
        # just storing this for later
        self.cleared_trades.append(msg)

    def handle_market_transaction(self, msg: PBMarketTransaction):
        """this tells the agent when it was able to buy something successfully"""
        self.purchases.append(msg)

    def send_order(self, action):
        order = PBOrder(broker=cfg.ME, timeslot=self._target_timeslot, mWh=action[0], limitPrice=action[1])
        log.debug("order ! {} mWh, -- {} $".format(order.mWh, order.limitPrice))
        # gotta send via dispatcher
        dispatcher.send(signals.OUT_PB_ORDER, msg=order)

    def handle_tariff_transaction(self, msg:PBTariffTransaction):
        self.realized_usage += msg.kWh / 1000
        pass


class PowerTacWholesaleObservation:
    """Helper class that wraps all the components of an observation that can be passed to the agent.
    Each agent implementation can decide what parts of this observation to make use of"""

    def __init__(self, hist_avg_prices: np.array,
                 step: int,
                 orderbooks: List[PBOrderbook],
                 purchases: List[np.array],
                 cleared_trades: List[PBClearedTrade],
                 predictions: List[float],
                 actions: List[np.array],
                 internals = None):
        self.hist_avg_prices: np.array = hist_avg_prices
        self.step = step
        self.orderbooks: List[PBOrderbook] = orderbooks
        self.purchases: List[PBMarketTransaction] = purchases
        self.cleared_trades: List[PBClearedTrade] = cleared_trades
        self.predictions: List[float] = predictions
        self.actions: List[np.array] = actions
        self.internals = internals #used by tensorforce


# not really needed anymore
# @deprecated
class WholesaleObservationSpace(spaces.Box):
    """
    - demand prediction - purchases 24x float
    - historical prices of currently traded TS 24x24 float (with diagonal TR-BL zeros in bottom right)
    - historical prices of last 168 timeslots
    - ... TODO more?
    """

    def __init__(self):
        # box needs min and max. using signed int32 min/max
        required_energy = Box(low=cfg.np_low, high=cfg.np_high, shape=(1,), dtype=np.float32)
        historical_prices = Box(low=cfg.np_low, high=cfg.np_high, shape=(168,), dtype=np.float32)
        current_prices = Box(low=cfg.np_low, high=cfg.np_high, shape=(24, 2), dtype=np.float32)
        super().__init__(low=cfg.np_low, high=cfg.np_high, shape=(1 + 24,), dtype=np.float32)


class WholesaleActionSpace(spaces.Box):
    """
    A wholesale action is anywhere in the 2D space with Dim 1 being limited to [-1,+1] and Dim 2 to [-2.0, +2.0]
        - Dimension 0 describes how much mWh to buy. Positive numbers buy, negative numbers sell The network gets fed a prediction regarding its portfolio balance.
          If the prediction is -243 (i.e. missing 243mWh for this timeslot to be matching the predicted demand) then
          - 0.5 means buying 243mWh.
          - 0 means buy nothing,
          - -0.5 means sell another 243mWh and
          - -1 means sell twice the predicted
          imbalance. If the prediction is +230 then 0.5 means buying another 230 and -0.5 means selling 230.
        - Dimension 1 describes the limit price. This is a mapping to the limit price.
          It's based on the known prices for the target timeslot.
             0 --> pay nothing
            -1 --> buy for 2x average known price
            +1 --> sell for 2x average known price
            +0.5 --> sell for average known price
            TODO is this maybe not large enough? increase box size for more freedom in pricing

    """

    def __init__(self):
        a = Box(low=np.array([-1.0, -1.0]), high=np.array([+1.0, +1.0]), dtype=np.float32)
        super().__init__(low=a.low, high=a.high, dtype=np.float32)
