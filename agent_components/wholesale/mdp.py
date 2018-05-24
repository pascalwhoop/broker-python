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
        timestep_actions = []
        for i in range(cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL):
            a = Box(low=np.array([-1.0, -2.0]), high=np.array([+1.0, +2.0]), dtype=np.float32)
            timestep_actions.append(a)
        super().__init__(tuple(timestep_actions))


class WholesaleObservationSpace(spaces.Dict):
    """
    - demand prediction - purchases 24x float
    - historical prices of currently traded TS 24x24 float (with diagonal TR-BL zeros in bottom right)
    - historical prices of last 168 timeslots
    - ... TODO more?
    """

    def __init__(self):
        # box needs min and max. using signed int32 min/max
        sizes = np.finfo(np.array([1.0], dtype=np.float32)[0])
        high = sizes.max
        low = sizes.min
        required_energy = Box(low=low, high=high, shape=(24,), dtype=np.float32)
        historical_prices = Box(low=low, high=high, shape=(168,), dtype=np.float32)
        current_prices = Box(low=low, high=high, shape=(24, 24, 2), dtype=np.float32)
        super().__init__({
            'required_energy': required_energy,
            'historical_prices': historical_prices,
            'current_prices': current_prices
        })


class FlatWholesaleObservationSpace(spaces.Box):
    def __init__(self):
        sizes = np.finfo(np.array([1.0], dtype=np.float32)[0])
        high = sizes.max
        low = sizes.min
        required_energy_length = cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL
        historical_length = cfg.WHOLESALE_HISTORICAL_DATA_LENGTH
        current_prices = cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL * cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL * 2
        all = required_energy_length + historical_length + current_prices
        super().__init__(low=low, high=high, shape=(all,), dtype=np.float32)


class PowerTacEnv(Env):
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


class PowerTacLogsMDPEnvironment(PowerTacEnv):
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
        self.num_envs = 1
        self.steps = 0
        self.action_space = WholesaleActionSpace()
        # self.observation_space = WholesaleObservationSpace()
        self.observation_space = FlatWholesaleObservationSpace()
        self.reward_range = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.wholesale_running_averages = {}
        self.forecasts = []  # holds the forecasts for the next 24 timeslots
        self.active_timeslots = deque(maxlen=cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL)
        self.historical_prices = deque(
            maxlen=cfg.WHOLESALE_HISTORICAL_DATA_LENGTH)  # holds the historical averages prices
        self.historical_prices.extend([0] * cfg.WHOLESALE_HISTORICAL_DATA_LENGTH)
        self.observations = []

        self.purchases = {}  # a map of purchases for each timeslot. timeslot --> list([mWh, price])

        # for mocking the market with the log files
        self._prices_files = get_wholesale_file_paths()
        self._usages_files = get_usage_file_paths()
        # self.initial_timeslot = 0
        self.wholesale_data = None
        self.demand_data = None  #

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

        # because the NN hands us a 1D array
        action = unflat_action(action)
        # translate into proper mWh/price actions
        real_actions = self.translate_action_to_real_world_vals(action)
        # get matching market data
        market_data = self.get_market_data_now()
        # evaluate cleared yes/no --> traded for cheaper? cleared (selling cheaper or buying more expensive)
        part_of_cleared = self.which_are_cleared(real_actions, market_data)

        # store the cleared trades --> those where the agent managed to be part of the clearing
        self.apply_clearings_to_purchases(real_actions, part_of_cleared, market_data)

        self.get_new_forecasts()
        self.append_historical_price()

        reward = self.calculate_reward()

        # ---------------------------------------------------------------
        # ------ stepping timeslot. Any methods that require
        # ------ the new data need to follow this breaking change in data
        # ---------------------------------------------------------------
        # deletes the first entry of the background data lists
        self._step_timeslot()
        # calculate reward for closed timestep
        observation = self.make_observation()
        flat_obs = make_flat_observation(observation)

        self.steps += 1
        done = False
        if self.steps > cfg.WHOLESALE_STEPS_PER_TRIAL:
            done = True
        return flat_obs, reward, done, {}

        # TODO calc reward
        # TODO done?
        # TODO info?

    def reset(self):
        """
        Overriding the Gym Env reset. In our trading world, there is no "reset" it's just a new prediction for a new timeslot
        :return:
        """
        self.steps = 0
        obs, r, done, info = self.step(get_do_nothing())
        return obs

    def apply_clearings_to_purchases(self, actions, cleared_list, market_data):
        # go over the active slots and place any purchases in the data
        assert len(actions) == len(cleared_list) == len(self.active_timeslots)

        for i, ts in enumerate(self.active_timeslots):
            if ts not in self.purchases:
                self.purchases[ts] = []

            if cleared_list[i]:
                purchase = [actions[i][0], market_data[i][1]]
                self.purchases[ts].append(purchase)

    def new_game(self):
        """Marks the environment as completed and therefore lets the agent learn again on a new game once it is ready"""
        # get new game number
        if not self.game_numbers:
            self.game_numbers = self._make_random_game_order()
        gn = self.game_numbers.pop()
        # getting data and storing it locally
        dd, wd = self.make_data_for_game(gn)
        self.wholesale_data: List = wd
        self.demand_data: List[float] = list(dd)
        self.get_new_forecasts()

        self.reset_active_timeslots(self.wholesale_data)
        self.observations = []

        # stepping the environment once, not doing anything
        obs, r, d, i =self.step(get_do_nothing())
        return obs

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
        idx = np.random.randint(0, high=len(demand), size=30)
        # using only random 30 picks from customers
        summed_random_30 = demand[idx, :].sum(axis=0)

        return self.trim_data(summed_random_30, wholesale_data, demand_data.get_first_timestep_for_file(usage_file))

    def _make_random_game_order(self):
        # whichever is shorter.
        max_game = len(self._prices_files) if len(self._prices_files) < len(self._usages_files) else len(
            self._usages_files)
        # mix up all the game numbers
        game_numbers = list(range(1, max_game))
        random.shuffle(game_numbers)
        return game_numbers

    def _step_timeslot(self):
        """Steps the game data up one notch. Removing first of most lists"""

        #not enough data left for this game? next game!
        if len(self.wholesale_data) < 25 or len(self.demand_data) < 25 :
            self.new_game()
            return

        # self.active_timeslots.append(self.active_timeslots[-1] + 1)
        self.active_timeslots.append(self.wholesale_data[24][0])
        self.wholesale_data.pop(0)  # removes current timeslot --> is realized now
        self.demand_data.pop(0)

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
            # data is a 24 item long array of 2 vals each
            # average is sum(price_i * kwh_i) / kwh_total
            sum_total = result[:, 0].sum()
            avg = 0
            if sum_total != 0.0:
                avg = (result[:, 0] * result[:, 1]).sum() / sum_total
            # if avg was not set or was set but to 0 use last average for this timestep
            if avg is 0 and averages:
                avg = averages[-1]
            averages.append(avg)

        return averages

    def get_current_knowledge_horizon(self):
        otp = cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL
        known_results = np.zeros((otp, otp, 2))
        for i, ts in enumerate(self.active_timeslots):
            row = self.wholesale_data[i]
            # we are using only the wholesale data mwh/price (starts at index 3) and then only a subset of the data
            # the data holds the historical data (i.e. everything about the target timeslot but we want to only use
            # the data up to the "now" i.e. not future averages
            assert ts == row[0]

            data = np.array(row[3:])[:23 - i]
            known_results[i, :23 - i] = data
        return known_results

    def translate_action_to_real_world_vals(self, action):
        action = np.array(action)
        known_results = self.get_current_knowledge_horizon()
        averages = self.calculate_running_averages(known_results)
        # here is where the meat is. first, amplify the action by 2 --> +1 == x2, -1 == x-2
        action = action * 2
        prices = action[:, 1] * np.array(averages).transpose()
        amounts = action[:, 0] * np.array(self.forecasts).transpose()
        real_actions = np.stack([amounts, prices], axis=1)
        return np.around(real_actions, decimals=5)

    def get_market_data_now(self):
        """Returns the market data of the currently active 24 timeslots.
        It does not remove the latest active timeslot, that is up to the calling method
        to ensure so the next timestep works as expected"""
        data = []
        for i, ts in enumerate(self.active_timeslots):
            # ensuring we have the right timeslot
            assert self.wholesale_data[i][0] == ts
            trading_index = 23 - i + 3  # the trades in the data are sorted left-right t-24 -- t-1 ... +3 because of the header data
            data.append(self.wholesale_data[i][trading_index])
        return np.array(data)

    def which_are_cleared(self, real_actions, market_data):
        # ignoring amounts in offline learning files, just checking prices
        a = real_actions[:, 1]
        m = market_data[:, 1]
        z = np.zeros(a.shape)

        # it's greater not greater equal because we want to be on the one or the other side of the clearing price. not on spot
        asks = np.logical_and(np.greater(a, z), np.greater(m, a))
        bids = np.logical_and(np.greater(z, a), np.greater(a * -1, m))
        return np.logical_or(asks, bids)

    def get_sum_purchased_for_ts(self, ts):
        if ts in self.purchases and len(self.purchases[ts]) > 0:
            return np.array(self.purchases[ts])[:, 0].sum()
        else:
            return 0

    def get_new_forecasts(self):
        """
        Call before _step_timeslot
        :return:
        """
        if cfg.WHOLESALE_FORECASTS_TYPE == 'perfect':
            self.forecasts = self.demand_data[1:1 + cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL]

    def make_observation(self):
        obs = {}
        obs['required_energy'] = np.array(self.forecasts) - np.array(
            [self.get_sum_purchased_for_ts(ts) for ts in self.active_timeslots])
        obs['historical_prices'] = np.array(self.historical_prices)
        obs['current_prices'] = np.array(self.get_current_knowledge_horizon())
        self.observations.append(obs)
        return obs

    def append_historical_price(self):
        d = np.array(self.wholesale_data[0][3:])
        self.historical_prices.append((d[0] * d[1]).sum() / d[1].sum())

    def calculate_reward(self):
        """Gives back a relation between the average market price for the target timeslot and the average price the broker achieved"""
        trades = self.wholesale_data[0][3:]
        average_market = self.calculate_running_averages(np.array([trades]))[0]

        bought = self.purchases[self.active_timeslots[0]]

        # appending final balancing costs for broker for any missing energy
        if len(bought) == 0:
            balancing_needed = self.demand_data[0]
        else:
            balancing_needed = self.calculate_balancing_needed(np.array(bought), self.demand_data[0])
        du_trans = []
        if balancing_needed > 0:
            # being forced to buy for 5x the market price! try and get your kWh in ahead of time is what it learns
            du_trans = [balancing_needed, -1 * average_market * 5]
        if balancing_needed < 0:
            # getting only a 0.5 of what the normal market price was
            du_trans = [balancing_needed, 0.5 * average_market]  # TODO to config
        # TODO for now just a fixed punishment for every balanced mWh. Later maybe based on balancing stats data
        if du_trans:
            bought.append(du_trans)

        average_agent, stupid = self.average_price_for_power_paid(bought)
        # returning relation between the two.
        # Market trading is always in positive numbers (perspective of "price per kWh sold")
        # but the broker is from its perspective ( "price per kWh purchased" )

        # if the agent value is negative it means it bought more than it sold. In this case it needs to reverse the
        # relation for the reward
        if stupid:
            # broker traded really badly! It sold more energy than it bought and paid for it as well
            # giving it a negative reward
            return average_agent / average_market
        if average_agent < 0:
            return average_agent * (-1) / average_market
        else:
            return average_market / average_agent

    def average_price_for_power_paid(self, bought):
        """
        calculates the average price per mWh the agent paid in the market. If the agent actually sold more than
        bought, the value becomes negative
        :param bought:
        :return: average price, stupidBoolean --> stupid if it paid money to sell energy on average
        """
        # [:,0] is the mWh purchased and [:,1] is the average price for that amount
        total_purchased = 0
        total_paid = 0
        for t in bought:
            total_purchased += t[0]
            if t[0] > 0:
                # buy
                total_paid += t[0] * t[1] * (-1)
            if t[0] < 0:
                total_paid += t[0] * t[1]  # is negative, because t[0] is negative (-mWh)

        if total_paid < 0 and total_purchased < 0:
            return (-1) * total_paid / total_purchased, False
        if total_paid > 0 and total_purchased > 0:
            return total_paid / total_purchased, False
        else:
            return total_paid / total_purchased, True

    def calculate_balancing_needed(self, purchases: np.array, demand):
        """
        Determines the amount of balancing that is needed for the agents actions after it purchased for 24 times.

        :param purchases: array of purchases
        :param demand: demand for that timeslot
        :return: amount of additional mWh the agent requires for its portfolio.
        """
        sum_purchased = purchases.sum(axis=0)[0]
        # if demand > purchases --> positive, otherwise negative
        return demand - sum_purchased


def make_flat_observation(observation) -> np.array:
    obs = []
    obs.extend(observation['required_energy'])
    obs.extend(observation['historical_prices'])
    obs.extend(observation['current_prices'].flatten())
    return np.array(obs)


def unflat_action(action: np.array):
    return action.reshape(cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL, 2)


def get_do_nothing():
    return np.zeros((cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL, 2)).flatten()
