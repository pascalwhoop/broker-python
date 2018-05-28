import ast
import csv
import logging
import math
import os
import random
import time
from collections import deque
from operator import itemgetter
from shutil import rmtree
from typing import List

import numpy as np
from gym import spaces
from gym.core import Env
from gym.spaces import Box
from keras.metrics import MSE
from sklearn import preprocessing

import agent_components.demand.data as demand_data
from communication.grpc_messages_pb2 import PBClearedTrade, PBMarketTransaction
import util.config as cfg

# core component for the wholesale mdp process.

# long term: this should be able to both support decisions in an active competition as well as learn from an active
# competition and state files

# short term: just learning from state files, picking a competing broker from the state file that we wanna learn from


# should allow for several kinds of policy determining approaches --> DeepQ, PolicyGradient, ..
from util.learning_utils import get_wholesale_file_paths, get_usage_file_paths, TbWriterHelper

log = logging.getLogger(__name__)
sizes = np.finfo(np.array([1.0], dtype=np.float32)[0])
np_high = sizes.max
np_low = sizes.min


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


class WholesaleObservationSpace(spaces.Box):
    """
    - demand prediction - purchases 24x float
    - historical prices of currently traded TS 24x24 float (with diagonal TR-BL zeros in bottom right)
    - historical prices of last 168 timeslots
    - ... TODO more?
    """

    def __init__(self):
        # box needs min and max. using signed int32 min/max
       required_energy = Box(low=np_low, high=np_high, shape=(1,), dtype=np.float32)
       historical_prices = Box(low=np_low, high=np_high, shape=(168,), dtype=np.float32)
       current_prices = Box(low=np_low, high=np_high, shape=(24, 2), dtype=np.float32)
       super().__init__(low=np_low, high=np_high, shape=(1 + 168 + 24 * 2,), dtype=np.float32)




class PowerTacEnv(Env):
    def __init__(self):
        super().__init__()

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self, mode='logging'):
        # might implement later
        pass

    def close(self):
        raise NotImplementedError


class PowerTacMDPEnvironment(PowerTacEnv):
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


class PowerTacLogsMDPEnvironment(PowerTacEnv):
    """This class simulates a powertac trading environment but is based on logs of historical games.
    It assumes that the broker actions have no impact on the clearing price which is a reasonable estimation for any market
    that has a large enough volume in relation to the broker trading volume. Of course this does not apply once the broker is
    large enough to itself have an influence on the clearing prices. In PowerTAC, the broker wil actually have a significant impact
    on the prices. Therefore this is an optimistic first stage learner for the broker. It will allow it to learn a certain base
    amount but will underperform once it acts in a live environment.

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
        - predictions from the demand predictor or alternatively, a true prediction (i.e. the real value)
          or a noisy prediction where the noise may be adapted
        - historical market clearing prices
        - rewards based on reward calculation function

    """

    def __init__(self):
        """TODO: to be defined1. """
        super().__init__()
        #required by framework
        self.num_envs = 1
        self.action_space = WholesaleActionSpace()
        self.observation_space = WholesaleObservationSpace()
        self.reward_range = Box(low=np_low, high=np_high, shape=(1,), dtype=np.float32)

        # holds the historical averages prices
        #---------------------------------------------------------------
        # needs to be reset on next timeslot
        #---------------------------------------------------------------
        # a list of purchases for current timeslot --> list([mWh, price])
        # negative mWh are sales, positive mWh are purchases. Can be considered as "flow from broker"
        self.purchases = []
        #current number of steps in active timeslot
        self.steps = 0

        #---------------------------------------------------------------
        # base data to generate the mdp from. Needs to be reset on new game base data
        #---------------------------------------------------------------
        self.wholesale_averages = {}
        self.active_target_timeslot = 0 #needs to be incremented
        # self.initial_timeslot = 0
        self.wholesale_data = None
        self.demand_data = None

        #---------------------------------------------------------------
        # stays until training is completed
        #---------------------------------------------------------------
        # for mocking the market with the log files
        self._wholesale_files = get_wholesale_file_paths()
        self._demand_files = get_usage_file_paths()
        self.game_numbers = self._make_random_game_order()

        self.tb_writer_helper = TbWriterHelper('mdp_agent')


    def step(self, action: np.array):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (np.array): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self.tb_writer_helper.write_any(action[0], 'mwh')
        self.tb_writer_helper.write_any(action[1], 'price')

        #TODO trying direct mappings
        real_action = action
        # translate into proper mWh/price actions
        #real_action = self.translate_action_to_real_world_vals(action)
        # get matching market data
        market_data = self.get_market_data_now()
        # evaluate cleared yes/no --> traded for cheaper? cleared (selling cheaper or buying more expensive)
        cleared = self.is_cleared(real_action, market_data)
        if cleared:
            self.purchases.append(real_action)

        #logging answers every once in a while to see hwo the agent is deciding
        # ---------------------------------------------------------------
        # ------ stepping timeslot
        # ---------------------------------------------------------------
        self.steps += 1
        done = False
        if self.steps >= cfg.WHOLESALE_STEPS_PER_TRIAL:
            done = True
            self.log_actions(self.demand_data[0], market_data[1], self.get_sum_purchased_for_ts(), real_action[1])

        #reward = self.calculate_mock_reward(action)
        if done:
            # calculate reward for closed timestep
            reward = self.calculate_done_reward()
        else:
            #reward = self.calculate_step_reward()
            reward = 0

        #hacking reward to force it to stay small in the middle
        #TODO dirty trick
        if action[0] > 1 or action[1] > 1 or action[0] < -1 or action[1] < -1:
            reward = 0

        observation = self.make_observation()

        return observation, reward, done, {}
        #return observation, reward, done, {}

    def reset(self):
        """
        Resets the game, meaning it does the MDP for the next timestep.
        :return:
        """
        self.steps = 0
        # set next active time step
        self.active_target_timeslot += 1
        self.purchases = []

        if len(self.wholesale_data) <= 1 or len(self.demand_data) <= 0:
            self.new_game()
        else:
            # removing latest timeslot
            self.wholesale_data.pop(0)
            self.demand_data.pop(0)

        return self.make_observation()

    def new_game(self):
        """Marks the environment as completed and therefore lets the agent learn again on a new game once it is ready"""
        # get new game number
        if not self.game_numbers:
            self.game_numbers = self._make_random_game_order()
        gn = self.game_numbers.pop()
        # getting data and storing it locally
        dd, wd = self.make_data_for_game(gn)
        self.wholesale_data: List = wd
        self.apply_wholesale_averages(wd)

        self.demand_data: List[float] = list(dd)
        fc = self.get_new_forecast()

    def apply_wholesale_averages(self, wd):
        avgs = self.calculate_running_averages(_get_wholesale_as_nparr(wd))
        for i, ts in enumerate([row[0] for row in wd]):
            self.wholesale_averages[ts] = avgs[i]

    def make_data_for_game(self, i):
        wholesale_file_path = self._wholesale_files[i]
        demand_file_path = self._demand_files[i]
        with open(wholesale_file_path) as file:
            wholesale_data = self.parse_wholesale_file(file)
        # let's reuse this
        # resetting first
        demand_data.clear()
        demand_data.parse_usage_game_log(demand_file_path)
        demand = demand_data.get_demand_data_values()
        idx = np.random.randint(0, high=len(demand), size=30)
        # using only random 30 picks from customers
        summed_random_30 = demand[idx, :].sum(axis=0)

        return self.trim_data(summed_random_30, wholesale_data, demand_data.get_first_timestep_for_file(demand_file_path))

    def _make_random_game_order(self):
        # whichever is shorter.
        max_game = len(self._wholesale_files) if len(self._wholesale_files) < len(self._demand_files) else len(
            self._demand_files)
        # mix up all the game numbers
        game_numbers = list(range(1, max_game))
        random.shuffle(game_numbers)
        return game_numbers

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

    def trim_data(self, demand_data: np.array, wholesale_data: np.array, first_timestep_demand):
        min_dd = first_timestep_demand
        # the wholesale data holds 3 columns worth of metadata (slot, dayofweek,hourofday)
        ws_header = np.array([row[0:3] for row in wholesale_data])
        min_ws = int(ws_header[:, 0].min())
        # getting the first common ts
        starting_timeslot = min_dd if min_dd > min_ws else min_ws

        # trim both at beginning to ensure common starting TS
        if min_dd > min_ws:
            # trim ws
            wholesale_data = wholesale_data[min_dd - min_ws:]
        else:
            # trim other
            demand_data = demand_data[min_ws - min_ws:]

        # now trim both at end to ensure same length
        max_len = len(demand_data) if len(demand_data) < len(wholesale_data) else len(wholesale_data)
        demand_data = demand_data[:max_len - 1]
        wholesale_data = wholesale_data[:max_len - 1]
        return demand_data, wholesale_data

    def calculate_running_averages(self, known_results: np.array):
        """
        Calculates the running averages of all timeslots that are currently traded in wholesale.
        iterates over the active timeslots. If a timeslot has only 0 as trades, it takes the average of the previous timestep
        """
        averages = []

        for result in known_results:
            avg = self.calculate_running_average(result)
            if avg is 0 and averages:
                avg = averages[-1]
            averages.append(avg)

        return averages

    def calculate_running_average(self, timeslot_trading_data):
        # data is a 24 item long array of 2 vals each
        # average is sum(price_i * kwh_i) / kwh_total
        sum_total = timeslot_trading_data[:, 0].sum()
        avg = 0
        if sum_total != 0.0:
            avg = (timeslot_trading_data[:, 0] * timeslot_trading_data[:, 1]).sum() / sum_total
        # if avg was not set or was set but to 0 use last average for this timestep
        return avg

    def get_current_knowledge_horizon(self):
        """calculates the known horizon for the current timeslot"""
        # we are using only the wholesale data mwh/price (starts at index 3) and then only a subset of the data
        # the data holds the historical data (i.e. everything about the target timeslot but we want to only use
        # the data up to the "now" i.e. not future averages
        known_results = np.zeros((cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL, 2))
        data = np.array(self.wholesale_data[0][3:])
        for i in range(self.steps):
            known_results[i] = data[i]
        return known_results

    def translate_action_to_real_world_vals(self, action):
        known_results = self.get_current_knowledge_horizon()
        average = self.calculate_running_average(known_results)
        if average == 0:
            try:
                #tryig to use previous price average as average
                key = self.wholesale_data[0][0]
                average = self.wholesale_averages[key-1]
            except:
                average = 0

        # here is where the meat is.
        # first, amplify the action by 2 --> +1 == x2, -1 == x-2
        action = action * 2
        price = action[1] * average
        amount = action[0] * self.get_new_forecast()
        return np.array([amount, price])

    def get_market_data_now(self):
        """Returns the market data of the currently active timeslot and the current step
        """
        return np.array(self.wholesale_data[0][3:][self.steps])

    def is_cleared(self, action, market_data) -> bool:
        # if both positive, the agent it trying to pull a fast one. Nope
        if action[0] > 0 and action[1] > 0:
            return False
        # ignoring amounts in offline learning files, just checking prices
        a = action[1]
        m = market_data[1]
        z = np.zeros(a.shape)

        #market not active, nothing sold/bought
        if market_data[0] == 0:
            return False

        if a > 0:

            # selling for less -> cleared
            return a < abs(m)
        if a < 0:
            # buying for more -> cleared
            return abs(a) > m
        # default, didn't buy anything or no price
        return False


    def get_sum_purchased_for_ts(self) -> float:
        if not self.purchases:
            return 0
        return np.array(self.purchases)[:, 0].sum()

    def get_new_forecast(self):
        fc = self.demand_data[0]
        for i in range(cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL - self.steps):
            err_mult = random.uniform(-cfg.WHOLESALE_FORECAST_ERROR_PER_TS, cfg.WHOLESALE_FORECAST_ERROR_PER_TS)
            err_add = fc * err_mult
            fc += err_add
        return fc

    def make_observation(self) -> np.array:
        obs = []
        #returns positive --> need to purchase
        required_energy = self.calculate_missing_energy(self.get_sum_purchased_for_ts(), self.demand_data[0])
        obs.append(required_energy)
        hist_from = self.active_target_timeslot - 168 - (cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL - self.steps)
        hist_till = hist_from + cfg.WHOLESALE_HISTORICAL_DATA_LENGTH
        avgs = []
        for i in range(hist_from, hist_till):
            try:
                avgs.append(self.wholesale_averages[i])
            except:
                #not found. boh
                avgs.append(0)
        assert len(avgs) == cfg.WHOLESALE_HISTORICAL_DATA_LENGTH
        obs.extend(avgs)
        current_prices = np.array(self.get_current_knowledge_horizon()).flatten()
        obs.extend(current_prices)
        np_obs = np.array(obs)
        #np_obs = preprocessing.normalize(np_obs.reshape(1, -1)).flatten()
        return np_obs

    def calculate_done_reward(self):
        """Gives back a relation between the average market price for the target timeslot and the average price the broker achieved"""
        trades = self.wholesale_data[0][3:]
        average_market = self.calculate_running_averages(np.array([trades]))[0]

        bought = self.purchases

        # appending final balancing costs for broker for any missing energy
        if len(bought) == 0:
            balancing_needed = self.demand_data[0]
            energy_sum = 0
        else:
            energy_sum = self.get_sum_purchased_for_ts()
            balancing_needed = self.calculate_missing_energy(energy_sum, self.demand_data[0])
        du_trans = self.calculate_du_fee(average_market, balancing_needed)
        # TODO for now just a fixed punishment for every balanced mWh. Later maybe based on balancing stats data
        if du_trans:
            bought.append(du_trans)

        type_, average_agent = self.average_price_for_power_paid(bought)
        market_relative_prices = 0
        if type_ == 'ask':
            # broker is overall selling --> higher is better
            market_relative_prices = average_agent / average_market
        if type_ == 'bid':
            #broker is overall buyer --> lower is better
            market_relative_prices = average_market / average_agent

        #reward is made up of several components
        # priority one: getting the mWh that are predicted
        # priority number two: getting those cheaply

        #if balancing high --> ratio close to 1
        weight_balancing = abs(balancing_needed / energy_sum + balancing_needed)
        weight_price = 1 - weight_balancing
        return weight_balancing * (1/balancing_needed) + weight_price * market_relative_prices
        #return market_relative_prices - (balancing_needed/ (balancing_needed + energy_sum))


    def calculate_du_fee(self, average_market, balancing_needed):
        du_trans = []
        if balancing_needed > 0:
            # being forced to buy for 5x the market price! try and get your kWh in ahead of time is what it learns
            du_trans = [balancing_needed, -1 * average_market * 5]
        if balancing_needed < 0:
            # getting only a 0.5 of what the normal market price was
            du_trans = [balancing_needed, 0.5 * average_market]  # TODO to config
        if balancing_needed == 0:
            du_trans = [0,0]
        return du_trans

    def calculate_squared_diff(self, bought):
        mse_diff = 0
        if self.demand_data[0] and bought:
            bought_sum = np.array(bought)[:, 0].sum()
            mse_diff = (bought_sum - self.demand_data[0]) ** 2
        return mse_diff

    def average_price_for_power_paid(self, bought):
        """
        calculates the average price per mWh the agent paid in the market. If the agent actually sold more than
        bought, the value becomes negative
        :param bought:
        :return: average price, stupidBoolean --> stupid if it paid money to sell energy on average
        """
        # [:,0] is the mWh purchased and [:,1] is the average price for that amount
        total_energy = 0
        total_money = 0
        for t in bought:
            total_energy += t[0]
            total_money += abs(t[0]) * t[1]

        # if both positive --> broker got energy for free --> 0
        # if both negative --> broker paid and lost energy --> infinite
        if total_energy < 0 and total_money < 0:
            return 'bid',  np_high #max number possible. simulates infinite
        # broker lost energy --> sold it
        if total_energy < 0:
            return 'ask',  abs(total_money / total_energy)
        #broker purchased energy and still made profit
        if total_money > 0 and total_energy > 0:
            return 'bid', 0
        #broker purchased energy
        if total_energy > 0:
            return 'bid', abs(total_money / total_energy)

    def calculate_missing_energy(self, purchases: float, demand: float):
        """
        Determines the amount of balancing that is needed for the agents actions after it purchased for 24 times.

        :param purchases: array of purchases. Positive means purchases, negative means sales of mWh
        :param demand: demand for that timeslot. Negative means agent needs to buy
        :return: amount of additional mWh the agent requires for its portfolio.
        """
        # if demand > purchases --> positive, otherwise negative
        # *(-1) because balancing needed is supposed to be from perspective of "missing transaction" for broker
        return (-1) * (demand + purchases)

    def log_actions(self, demand_forecast, last_price, purchased_sum, price_offered):
        divergence = abs(purchased_sum / demand_forecast)
        price_diff = last_price - abs(price_offered)
        self.tb_writer_helper.write_any(divergence, 'divergence')
        self.tb_writer_helper.write_any(price_diff, 'price_diff')

def _get_wholesale_as_nparr(wholesale_data: List):
    """Assumes it's being passed a list of wholesale data, where the first three columns are metadata and then it's raw stuff"""
    return np.array([row[3:] for row in wholesale_data])

def make_flat_observation(observation) -> np.array:
    obs = []
    obs.extend(observation['required_energy'])
    obs.extend(observation['historical_prices'])
    obs.extend(observation['current_prices'].flatten())
    return np.array(obs)


def unflat_action(action: np.array):
    return action.reshape(cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL, 2)


def get_do_nothing():
    return np.array([0,0])


