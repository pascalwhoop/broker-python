"""
This is a version 2 of the log based learning. First, I just patched together a RL Env from the log data. This one actually
emulates all the necessary events coming in from the server and the estimator component. It sends the messages
via pydispatch and then just goes through the usual infrastructure (EnvManager -> Env -> Agent).
"""

import logging
import numpy as np
import random
from gym.spaces import Box
from pydispatch import dispatcher
from typing import List

from agent_components.demand.estimator import CustomerPredictions
from agent_components.demand.learning import data as demand_data
from agent_components.wholesale.environments.PowerTacEnv import PowerTacWholesaleAgent, WholesaleActionSpace, \
    WholesaleEnvironmentManager, WholesaleObservationSpace
from agent_components.wholesale.util import parse_wholesale_file
from communication.grpc_messages_pb2 import PBClearedTrade, PBMarketTransaction, PBOrder, PBTimeslotComplete, \
    PBTimeslotUpdate, PBTariffTransaction, PBTxType, PRODUCE, CONSUME
from communication.pubsub import signals
from communication.pubsub.PubSubTypes import SignalConsumer
from util import config as cfg
from util.learning_utils import get_usage_file_paths, get_wholesale_file_paths

log = logging.getLogger(__name__)


class LogEnvManagerAdapter(SignalConsumer):
    """This class simulates a powertac trading environment but is based on logs of historical games.  It assumes that
    the broker actions have no impact on the clearing price which is a reasonable estimation for any market that has a
    large enough volume in relation to the broker trading volume. Of course this does not apply once the broker is large
    enough to itself have an influence on the clearing prices. In PowerTAC, the broker wil actually have a significant
    impact on the prices. Therefore this is an optimistic first stage learner for the broker. It will allow it to learn
    a certain base amount but will underperform once it acts in a live environment.

    The basic skills the broker learns in the wholesale trading are as follows:

    - based on a (changing) demand forecast, try to equalize the portfolio so that the broker doesn't incur any
      balancing costs by the DU
    - try to pay as little as possible for the energy needed at timeslot x. Buying earlier is cheaper but riskier

    These two goals are reasoned by the following assumptions: The wholesale trader has no influence on the amount of
    energy needed by its customers.  This is a partial truth because some brokers may be able to curtail their customers
    usage if market prices are too high and the cost of curtailing the customer is valued less than the cost of
    purchasing and delivering the energy. Because the current broker implementation does not make use of this ability,
    the assumption is correct.  Another assumption is the idea of the agents actions not influencing the clearing price.
    The server logs suggest clearing amounts of low two digit megawatt per timeslot.  If the broker simply tries to
    predict small amounts of energy, this assumption is appropriate. A broker that only represents a few dozen private
    households would therefore trade small kilowatt amounts per timeslot, barely influencing the market prices. An
    on-policy RL agent may therefore still learn successfully, despite the fact that the environment doesn't *actually*
    react to its actions.

    To allow the broker to learn with offline files, the following process is taken:

    - Creation of market price statistics with the `org.powertac.logtool.example.MktPriceStats` class
    - Creation of usage data with the `org.powertac.logtool.example.CustomerProductionConsumption` class
    - selecting a small set of customers as a permanent customer portfolio for the broker
    - passing observations to the agent
        - predictions from the demand predictor or alternatively, a true prediction (i.e. the real value) or a noisy
          prediction where the noise may be adapted
        - historical market clearing prices
        - rewards based on reward calculation function

    """

    def __init__(self, agent: PowerTacWholesaleAgent):

        # handling params
        self.agent = agent
        self.games_played = 0
        self.env_manager: WholesaleEnvironmentManager = None
        self.step_rewards = 0
        # required by framework
        self.action_space = WholesaleActionSpace()
        self.observation_space = WholesaleObservationSpace()
        self.reward_range = Box(low=cfg.np_low, high=cfg.np_high, shape=(1,), dtype=np.float32)

        self.orders: List[PBOrder] = []

        # current timestep. the next X are open for trading (X set in config)
        self.current_timestep = 0
        self.first_timestep = self.current_timestep

        # ---------------------------------------------------------------
        # base data to generate the mdp from. Needs to be reset on new game base data
        # ---------------------------------------------------------------
        # self.initial_timeslot = 0
        self.wholesale_data = {}
        self.demand_data = None  # careful here, its kWh! #TODO make mWh

        # ---------------------------------------------------------------
        # stays until training is completed
        # ---------------------------------------------------------------
        # for mocking the market with the log files
        self._wholesale_files = get_wholesale_file_paths()
        self._demand_files = get_usage_file_paths()
        self.game_numbers = self._make_random_game_order()

    def subscribe(self):
        # need to catch all orders and determine if they lead to clearing
        dispatcher.connect(self.handle_order, signal=signals.OUT_PB_ORDER)

    def unsubscribe(self):
        dispatcher.disconnect(self.handle_order, signal=signals.OUT_PB_ORDER)

    def handle_order(self, sender, signal, msg: PBOrder):
        self.orders.append(msg)

    def start(self, max_games=None):
        """Starts the learning, with control coming 'from the server', not from the agent"""
        # take random game
        # start stepping through timeslots, 24 at a time
        while self.game_numbers:
            #pops one from numbers
            self.new_game()
            self.current_timestep = self.get_first_timestep()
            self.first_timestep = self.current_timestep
            #create env_manager
            self.env_manager = WholesaleEnvironmentManager(self.agent)
            self.env_manager.subscribe()
            self.step_game()
            #while self.current_timestep < self.wholesale_data
            self.games_played +=1
            if max_games and max_games < self.games_played:
                break


    def step_game(self):
        """loop per game. simulates all events coming from server and by listening to the PBOrder events,
        responds to agent actions"""
        while self.wholesale_data:
            #evaluate any orders received in previous step and send PBMarketTransaction
            self.evaluate_orders_received()
            #normally triggers demand forecasting --> predictions
            self.simulate_timeslot_complete()
            # ----- Timeslot cut -----
            #send out Transactions by customers
            self.simulate_tariff_transactions()
            #send out PBTimeslotUpdate
            self.simulate_timeslot_update()
            #send out Predictions based on DemandData
            self.simulate_predictions()
            #send out PBClearedTrade for the next 24h timesteps
            self.simulate_cleared_trade()

            # the stepping is sort of "half dependent" on previous data.
            # in Timestep 363, ClearedTrades and MarketTransactions that refer to 362 are given out.
            # therefore at THE END of the step, the PREVIOUS timestep data is deleted
            if self.current_timestep-1 in self.wholesale_data:
                del self.wholesale_data[self.current_timestep-1]
            self.current_timestep+=1

    def new_game(self):
        """load data for new game into object"""
        # get new game number
        if not self.game_numbers:
            self.game_numbers = self._make_random_game_order()
        if hasattr(cfg, 'WHOLESALE_OFFLINE_TRAIN_GAME'):
            # only using this one game!
            gn = cfg.WHOLESALE_OFFLINE_TRAIN_GAME
        else:
            gn = self.game_numbers.pop()
        # getting data and storing it locally
        self.make_data_for_game(gn)

    def make_data_for_game(self, i):
        wholesale_file_path = self._wholesale_files[i]
        self.make_wholesale_data(wholesale_file_path)

        demand_file_path = self._demand_files[i]
        self.make_demand_data(demand_file_path)

    def make_wholesale_data(self, wholesale_file_path):
        with open(wholesale_file_path) as file:
            wholesale_data = parse_wholesale_file(file)
        for ts in wholesale_data:
            timestep = ts[0]
            data = np.array(ts[3:])
            self.wholesale_data[timestep] = data

    def make_demand_data(self, demand_file_path):
        # let's reuse this
        # resetting first
        demand_data.clear()
        # getting unscaled predictions
        demand_data.parse_usage_game_log(demand_file_path, pp_type='none')
        demand = demand_data.get_demand_data_values()
        # using only random 30 picks from customers
        idx = np.random.randint(0, high=len(demand), size=30)
        demand = demand[idx, :]

        #brutally trimming down all demand by 1/1000. It's irrelevant how much the forecast is, the agent needs to learn
        #to order what is required of it
        demand = demand /1000

        self.demand_data = demand

    def _make_random_game_order(self):
        # whichever is shorter.
        max_game = len(self._wholesale_files) if len(self._wholesale_files) < len(self._demand_files) else len(
            self._demand_files)
        game_numbers = list(range(1, max_game))
        if cfg.WHOLESALE_OFFLINE_TRAIN_RANDOM_GAME:
            # mix up all the game numbers
            # for reproducability and comparability, only shuffling when set in config
            random.shuffle(game_numbers)
        return game_numbers

    def get_first_timestep(self):
        return np.array(list(self.wholesale_data.keys())).min()

    def simulate_cleared_trade(self):
        """simulates the sending of PBClearedTrade messages for the next [t-1,t-1+24] timesteps"""
        last_step = self.current_timestep -1
        cleared_steps = list(range(last_step, last_step+ cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL))
        for i, s in enumerate(cleared_steps):
            if s not in self.wholesale_data:
                break
            data = self.wholesale_data[s]
            cleared_data = data[i]
            trade = PBClearedTrade(timeslot=s, executionMWh=cleared_data[0], executionPrice=cleared_data[1])
            dispatcher.send(signals.PB_CLEARED_TRADE, msg=trade)

    def simulate_timeslot_update(self):
        """Simulates the TimeslotUpdate message"""
        now = self.current_timestep
        dispatcher.send(signals.PB_TIMESLOT_UPDATE, msg=PBTimeslotUpdate(firstEnabled=now, lastEnabled=now+cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL))

    def evaluate_orders_received(self):
        """Evaluate order and check if it should be cleared"""
        cleared_mask = []
        for o in self.orders:
            #ignore orders at the end of a game
            if o.timeslot not in self.wholesale_data:
                continue
            distance = o.timeslot - self.current_timestep
            ts_data = self.wholesale_data[o.timeslot]
            market_clearing = ts_data[distance]
            cleared, prob = self.is_cleared_with_volume_probability(o, market_clearing)
            cleared_mask.append((cleared, distance))
            if cleared:
                #price is positive only when mWh is smaller 0
                price = market_clearing[1] * -1 if o.mWh > 0 else market_clearing[1]
                volume_received = o.mWh * prob #assuming we only get a part of what we want
                #sending out message
                dispatcher.send(signal=signals.PB_MARKET_TRANSACTION,
                                msg=PBMarketTransaction(price=market_clearing[1],
                                                        mWh=volume_received,
                                                        timeslot=o.timeslot))
            else:
                #not cleared
                pass
        log.info("Cleared count: " + ' '.join([str(i[1]) for i in cleared_mask if i[0]]))
        self.orders = []

    def is_cleared_with_volume_probability(self, order:PBOrder, marketClearing) -> (bool, float):
        """clears the order with a high probability if the offer was very generous and less prob if it is close to the market price"""
        action = np.array([order.mWh, order.limitPrice])
        clearing_probability = 1

        # if both positive, the agent it trying to pull a fast one. Nope
        if action[0] > 0 and action[1] > 0:
            #log.warning("agent is returning irrational bids")
            return False, 0
        if action[0] < 0 and action[1] < 0:
            #log.warning("agent is giving away money")
            return False, 0

        market_volume = abs(marketClearing[0])
        action_volume  = abs(action[0])
        if market_volume == 0:
            return False, 0
        if action_volume > market_volume:
            log.warning("agent is asking for large volume {} in market {}".format(action_volume, market_volume))
            return False, 0
        volume_diff = market_volume - action_volume
        #the larger the action_volume, the less likely that the market will clear it.
        clearing_probability *=  (volume_diff / market_volume)


        action_price = action[1]
        #TODO can I do this? are there negative market prices? shouldn't be but who knows
        market_price = abs(marketClearing[1])

        #selling energy
        if action_price >= 0:
            # selling for less -> cleared
            price_diff = market_price - action_price
        else:
            # buying for more -> cleared
            price_diff = abs(action_price) - market_price

        if price_diff <= 0:
            #buying too cheap or selling to expensive
            return False, 0
        price_multiplicator = (price_diff / market_price)
        price_multiplicator = np.array([price_multiplicator, 1]).min()
        clearing_probability *= price_multiplicator

        return random.random() < clearing_probability, clearing_probability

    def simulate_timeslot_complete(self):
        dispatcher.send(signals.PB_TIMESLOT_COMPLETE, PBTimeslotComplete(timeslotIndex=self.current_timestep-1))

    def simulate_predictions(self):
        dd = self.demand_data
        fts = self.first_timestep
        start = self.current_timestep+1
        end = start + cfg.DEMAND_FORECAST_DISTANCE
        demand_data = dd[:,start-fts:end-fts]
        demand_data = demand_data / 1000 #dividing by 1000 to turn kWh into mWh
        if cfg.WHOLESALE_FORECAST_ERROR_PER_TS > 0:
            demand_data = np.array([self.fuzz_forecast_for_training(customer_data) for customer_data in demand_data])
        preds = []
        for cust_number, customer_data in enumerate(demand_data):
            customer_pred_obj = CustomerPredictions("customer{}".format(cust_number), predictions=customer_data, first_ts=start)
            preds.append(customer_pred_obj)
        dispatcher.send(signals.COMP_USAGE_EST, msg=preds)

    def fuzz_forecast_for_training(self, customer_data):
        """fuzzes the data a bit according to configured noise factor"""
        for i in range(len(customer_data)):
            err_mult = random.uniform(-cfg.WHOLESALE_FORECAST_ERROR_PER_TS, cfg.WHOLESALE_FORECAST_ERROR_PER_TS)
            # fuzz them increasingly much
            customer_data[i:] = customer_data[i:] + err_mult * customer_data[i:]
        return customer_data

    def simulate_tariff_transactions(self):
        """sends out simulated timeslot """
        timestep = self.current_timestep - self.first_timestep
        if len(self.demand_data)< timestep:
            return
        usages = self.demand_data[:, timestep]
        for u in usages:
            if u < 0:
                t= CONSUME
            else:
                t = PRODUCE
            dispatcher.send(signals.PB_TARIFF_TRANSACTION, msg=PBTariffTransaction(txType=t, kWh=u, postedTimeslot=self.current_timestep))
