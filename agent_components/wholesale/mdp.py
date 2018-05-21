import ast
import csv
import random
from collections import deque
from typing import List

import numpy as np
from gym import spaces
from gym.core import Env
from gym.spaces import Box

import agent_components.demand.data as demand_data
from communication.grpc_messages_pb2 import PBClearedTrade, PBMarketTransaction
import util.config as cfg

# core component for the wholesale mdp process.

# long term: this should be able to both support decisions in an active competition as well as learn from an active
# competition and state files

# short term: just learning from state files, picking a competing broker from the state file that we wanna learn from


# should allow for several kinds of policy determining approaches --> DeepQ, PolicyGradient, ..
from util.learning_utils import get_wholesale_file_paths, get_usage_file_paths


class WholesaleActionSpace(spaces.Tuple):
    """
    A wholesale action is anywhere in the 2D space with Dim 1 being limited to [-1,+1] and Dim 2 to [-2.0, +2.0]
        - Dimension 0 describes how much kWh to buy. Positive numbers buy, negative numbers sell The network gets fed a prediction regarding its portfolio balance.
          If the prediction is -243 (i.e. missing 243kWh for this timeslot to be matching the predicted demand) then
          - 0.5 means buying 243kWh.
          - 0 means buy nothing,
          - -0.5 means sell another 243kWh and
          - -1 means sell twice the predicted
          imbalance. If the prediction is +230 then 0.5 means buying another 230 and -0.5 means selling 230.
        - Dimension 1 describes the limit price. This is a mapping to the limit price.
          It's based on the known prices for the target timeslot.
             0 --> pay nothing
            -1 --> buy for 2x average known price
            +1 --> sell for 2x average known price
            +0.5 --> sell for average known price

    """

    def __init__(self):
        timestep_actions = []
        for i in range(cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL):
            a = Box(low=np.array([-1.0, -2.0]), high=np.array([+1.0, +2.0]), dtype=np.float32)
            timestep_actions.append(a)
        super().__init__(tuple(timestep_actions))


class PowerTacMDPEnvironment(Env):
    """This class creates an adapter between the OpenAI Env class and the powertac environment where a RL agent performs
    the wholesale trading. Each timeslot is considered a distinct environment and the agent performs 24 steps before
    arriving at the terminal t-0 state. 
    
    There are a couple of things to be aware of:
        - PowerTAC has its own time. If the agent doesn't do anything or doesn't decide fast enough, the server doesn't
          care. 
        - OpenAI Gym doesn't mind waiting and the agent is the entitiy that decides when the next step occurs.
        - This means there is some "reversing" required in this class. 
            - Hence the step class blocks until an answer has been received by the server. 
            - If the agent takes too long to decide, the next server state is 
    """

    def __init__(self, target_timestep):
        """TODO: to be defined1. """
        Env.__init__(self)

        # powertac specifics
        self.target_timestep = target_timestep
        self.cleared_trades: List[PBClearedTrade] = []
        self.transactions: List[PBMarketTransaction] = []
        self.forecasts: List[demand_data.DemandForecasts]

    def update_forecasts(self, fc: demand_data.DemandForecasts):
        """
        Append new known forecasts to our history of knowledge about the world. 
        """
        self.forecasts.append(fc)

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        # pass action to server
        # block until answers are all collected
        # return with observation (TODO build this), reward (TODO calc) and done if reached terminal state

        pass

    def reset(self) -> None:
        """Marks the environment as completed and therefore lets the agent learn again on a new timeslot once it is ready"""
        pass

    def render(self, mode='logging') -> None:
        """
        Nothing to render. Although  may log or output some stuff to tensorboard. TBD
        """
        pass

    def close(self) -> None:
        """
        May be used for cleaning up things. Not needed now
        """
        pass


class PowerTacLogsMDPEnvironment(Env):
    """This class simulates a powertac trading environment but is based on logs of historical games.
    It assumes that the broker actions have no impact on the clearing price which is a reasonable estimation for any market
    that has a large enough volume in relation to the broker trading volume. Of course this does not apply once the broker is
    large enough to itself have an influence on the clearing prices.

    The basic skills the broker learns in the wholesale trading are as follows:

    - based on a (changing) demand forecast, try to equalize the portfolio so that the broker doesn't incur any balancing costs by the DU
    - try to pay as little as possible for the energy needed at timeslot x. Buying earlier is cheaper but riskier

    These two goals are reasoned by the following assumptions: The wholesale trader has no influence on the amount of energy needed by its customers.
    This is a partial truth because some brokers may be able to curtail their customers usage if market prices are too high and the cost of curtailing the customer
    is valued less than the cost of purchasing and delivering the energy. Because the current broker implementation does not make use of this ability, the assumption is correct.
    Another assumption is the idea of the agents actions not influencing the clearing price. The server logs suggest clearing amounts of low two digit megawatt per timeslot.
    If the broker simply tries to predict small amounts of energy, this assumption is appropriate. A broker that only represents a few dozen private households would therefore
    trade small kilowatt amounts per timeslot, barely influencing the market prices. An on-policy RL agent may therefore still learn successfully, despite
    the fact that the environment doesn't *actually* react to its actions.

    To allow the broker to learn with offline files, the following process is taken:

    - Creation of market price statistics with the `org.powertac.logtool.example.MktPriceStats` class
    - Creation of usage data with the `org.powertac.logtool.example.CustomerProductionConsumption` class
    - selecting a small set of customers as a permanent customer portfolio for the broker
    - passing observations to the agent
        - predictions from the demand predictor
        - historical market clearing prices
        - rewards based on reward calculation function

    """

    def __init__(self):
        """TODO: to be defined1. """
        Env.__init__(self)
        self.action_space = WholesaleActionSpace()
        #TODO define self.observation_space
        #TODo define self.reward_range

        self.wholesale_running_averages = {}
        self.forecasts = [] # holds the forecasts for the next 24 timeslots
        self.active_timeslots = deque(maxlen=24)

        self.purchases = {} #a map of purchases for each timeslot. timeslot --> list([mWh, price])

        # for mocking the market with the log files
        self._prices_files = get_wholesale_file_paths()
        self._usages_files = get_usage_file_paths()
        #self.initial_timeslot = 0
        self.wholesale_data = None
        self.demand_data = None #

        self.game_numbers = self._make_random_game_order()
        # iterate over past games randomly
        # this needs to be moved out of this.
        # step needs to be able to receive the next 24h timesteps. If there aren't enough left, the next game needs to be loaded
        # and the next games data is returned.
        # probably with a game_numbers.pop() approach
        # so -->
        # if has_steps():
        #     get steps
        # else:
        #     reset() --> new game loaded and observations returned

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        # translate into proper mWh/price actions
        real_actions = self.translate_action_to_real_world_vals(action)
        # get matching market data
        market_data = self.get_market_data_now()
        # evaluate cleared yes/no --> traded for cheaper? cleared (selling cheaper or buying more expensive)
        part_of_cleared = self.which_are_cleared(real_actions, market_data)

        #store the cleared trades --> those where the agent managed to be part of the clearing
        self.apply_clearings_to_purchases(real_actions, part_of_cleared)

        # calculate kWh "have" --> what's left
        # calculate reward for closed timestep
        # block until answers are all collected
        # return with observation (TODO build this), reward (TODO calc) and done if reached terminal state

        pass

    def apply_clearings_to_purchases(self, actions, cleared_list):
        #go over the active slots and place any purchases in the data
        assert len(actions) == len(cleared_list) == len(self.active_timeslots)

        for i, ts in enumerate(self.active_timeslots):
            if ts not in self.purchases:
                self.purchases[ts] = []

            if cleared_list[i]:
                self.purchases[ts].append(actions[i])

    def reset(self):
        """Marks the environment as completed and therefore lets the agent learn again on a new game once it is ready"""
        #get new game number
        if not self.game_numbers:
            self.game_numbers = self._make_random_game_order()
        gn = self.game_numbers.pop()
        #getting data and storing it locally
        dd, wd = self.make_data_for_game(gn)
        self.wholesale_data = wd
        self.demand_data = dd

        self.reset_active_timeslots(self.wholesale_data)

        #stepping the environment once, not doing anything
        return self.step(get_do_nothing())

    def reset_active_timeslots(self, wholesale_data):
        # setting new timeslots
        self.active_timeslots.clear()
        for i in range(cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL):
            self.active_timeslots.append(wholesale_data[i][0])

    def make_data_for_game(self, i):
        price_file = self._prices_files[i]
        usage_file = self._usages_files[i]
        with open(price_file) as file:
            wholesale_data = self.parse_wholesale_file(file)
        # let's reuse this
        # resetting first
        demand_data.clear()
        demand_data.parse_usage_game_log(usage_file)
        demand = demand_data.get_demand_data_values()
        summed_first_30 = demand[:30,:].sum(axis=0)

        return self.trim_data(summed_first_30, wholesale_data, demand_data.get_first_timestep_for_file(usage_file))

    def _make_random_game_order(self):
        # whichever is shorter.
        max_game = len(self._prices_files) if len(self._prices_files) < len(self._usages_files) else len(
            self._usages_files)
        # mix up all the game numbers
        game_numbers = list(range(1, max_game))
        random.shuffle(game_numbers)
        return game_numbers


    def _step_timeslot(self):
        self.active_timeslots.append(self.active_timeslots[-1] + 1)


    def render(self, mode='logging') -> None:
        """
        Nothing to render. Although  may log or output some stuff to tensorboard. TBD
        """
        pass

    def close(self) -> None:
        # TODO close the environment, we're done here!
        pass

    def parse_wholesale_file(self, file):
        out = []
        reader = csv.reader(file)
        for row in reader:
            out.append([ast.literal_eval(str.strip(cell).replace(' ', ',')) for cell in row])
        return out

    def trim_data(self, demand_data:np.array, wholesale_data: np.array, first_timestep_demand):
        min_dd = first_timestep_demand
        # the wholesale data holds 3 columns worth of metadata (slot, dayofweek,hourofday)
        ws_header = np.array([row[0:3] for row in wholesale_data])
        min_ws = int(ws_header[:,0].min())
        #getting the first common ts
        starting_timeslot = min_dd if min_dd > min_ws else min_ws

        #trim both at beginning to ensure common starting TS
        if min_dd > min_ws:
            #trim ws
            wholesale_data = wholesale_data[min_dd-min_ws:]
        else:
            #trim other
            demand_data = demand_data[min_ws-min_ws:]

        #now trim both at end to ensure same length
        max_len = len(demand_data) if len(demand_data) < len(wholesale_data) else len(wholesale_data)
        demand_data = demand_data[:max_len-1]
        wholesale_data = wholesale_data[:max_len-1]
        return demand_data, wholesale_data

    def calculate_running_averages(self):
        """
        Calculates the running averages of all timeslots that are currently traded in wholesale.
        iterates over the active timeslots. If a timeslot has only 0 as trades, it takes the average of the previous timestep
        """
        averages = []
        for i, ts in enumerate(self.active_timeslots):
            row = self.wholesale_data[i]
            #we are using only the wholesale data mwh/price (starts at index 3) and then only a subset of the data
            #the data holds the historical data (i.e. everything about the target timeslot but we want to only use
            # the data up to the "now" i.e. not future averages
            assert ts == row[0]
            data = np.array(row[3:])[:23-i]
            #data is a 24 item long array of 2 vals each
            #average is sum(price_i * kwh_i) / kwh_total
            sum_total = data[:, 0].sum()
            avg = 0
            if sum_total != 0.0:
                avg = (data[:,0] * data[:,1]).sum() / sum_total
            # if avg was not set or was set but to 0 use last average for this timestep
            if avg is 0 and averages:
                avg = averages[-1]
            averages.append(avg)
        return averages

    def translate_action_to_real_world_vals(self, action):
        action = np.array(action)
        averages = self.calculate_running_averages()
        # here is where the meat is. first, amplify the action by 2 --> +1 == x2, -1 == x-2
        action  = action * 2
        prices  = action[:,1] * np.array(averages).transpose()
        amounts = action[:,0] * np.array(self.forecasts).transpose()
        real_actions = np.stack([amounts, prices], axis=1)
        return np.around(real_actions, decimals=5)

    def get_market_data_now(self):
        """Returns the market data of the currently active 24 timeslots.
        It does not remove the latest active timeslot, that is up to the calling method
        to ensure so the next timestep works as expected"""
        data = []
        for i,ts in enumerate(self.active_timeslots):
            #ensuring we have the right timeslot
            assert self.wholesale_data[i][0] == ts
            trading_index = 23-i+3 #the trades in the data are sorted left-right t-24 -- t-1 ... +3 because of the header data
            data.append(self.wholesale_data[i][trading_index])
        return np.array(data)

    def which_are_cleared(self, real_actions, market_data):
        # ignoring amounts in offline learning files, just checking prices
        a = real_actions[:, 1]
        m = market_data[:,  1]
        z = np.zeros(a.shape)

        # it's greater not greater equal because we want to be on the one or the other side of the clearing price. not on spot
        asks = np.logical_and(np.greater(a, z), np.greater(m, a))
        bids = np.logical_and(np.greater(z, a), np.greater(a*-1, m))
        return np.logical_or(asks, bids)






def get_do_nothing():
    action = []
    for i in range(24):
        a = Box(low=np.array([-0.0, -0.0]), high=np.array([+0.0, +0.0]),dtype=np.float32)
        action.append(a)
    return tuple(action)
