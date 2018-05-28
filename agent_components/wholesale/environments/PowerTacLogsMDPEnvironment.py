import random
from typing import List

import numpy as np
from gym.spaces import Box

from agent_components.demand import data as demand_data
from agent_components.wholesale.learning.reward_functions import simple_truth_ordering
from agent_components.wholesale.mdp import WholesaleActionSpace, WholesaleObservationSpace, np_low, \
    np_high, _get_wholesale_as_nparr, parse_wholesale_file, price_scaler, demand_scaler
from agent_components.wholesale.environments.PowerTacEnv import PowerTacEnv
from agent_components.wholesale.util import calculate_running_averages, calculate_missing_energy, trim_data, is_cleared
from util import config as cfg
from util.learning_utils import get_wholesale_file_paths, get_usage_file_paths, TbWriterHelper

tb_writer_helper = TbWriterHelper('mdp_agent')


class PowerTacLogsMDPEnvironment(PowerTacEnv):
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

    def __init__(self, reward_func=simple_truth_ordering):
        """TODO: to be defined1. """
        super().__init__()

        # handling params
        self.calculate_reward = reward_func
        # required by framework
        self.num_envs = 1
        self.action_space = WholesaleActionSpace()
        self.observation_space = WholesaleObservationSpace()
        self.reward_range = Box(low=np_low, high=np_high, shape=(1,), dtype=np.float32)

        # holds the historical averages prices
        # ---------------------------------------------------------------
        # needs to be reset on next timeslot
        # ---------------------------------------------------------------
        # a list of purchases for current timeslot --> list([mWh, price])
        # negative mWh are sales, positive mWh are purchases. Can be considered as "flow from broker"
        self.purchases = []
        # current number of steps in active timeslot
        self.steps = 0
        self.latest_observeration = None

        # ---------------------------------------------------------------
        # base data to generate the mdp from. Needs to be reset on new game base data
        # ---------------------------------------------------------------
        self.wholesale_averages = {}
        self.active_target_timeslot = 0  # needs to be incremented
        # self.initial_timeslot = 0
        self.wholesale_data = None
        self.demand_data = None

        # ---------------------------------------------------------------
        # stays until training is completed
        # ---------------------------------------------------------------
        # for mocking the market with the log files
        self._wholesale_files = get_wholesale_file_paths()
        self._demand_files = get_usage_file_paths()
        self.game_numbers = self._make_random_game_order()

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
        tb_writer_helper.write_any(action[0], 'mwh')
        tb_writer_helper.write_any(action[1], 'price')

        # translate into proper mWh/price actions
        real_action = self.translate_action_to_real_world_vals(action)
        # get matching market data
        market_data = self.get_market_data_now()
        # evaluate cleared yes/no --> traded for cheaper? cleared (selling cheaper or buying more expensive)
        cleared = is_cleared(real_action, market_data)
        if cleared:
            self.purchases.append(real_action)

        # logging answers every once in a while to see hwo the agent is deciding
        # ---------------------------------------------------------------
        # ------ stepping timeslot
        # ---------------------------------------------------------------
        self.steps += 1
        reward = 0
        done = self.is_done(market_data, real_action)

        if done:
            # calculate reward for closed timestep
            reward = self.calculate_reward(action, self.wholesale_data[0][3:], self.purchases,
                                           self.get_forecast_for_active_ts())

        observation = self.make_observation()

        return observation, reward, done, {}
        # return observation, reward, done, {}

    def is_done(self, market_data, real_action):
        done = False
        if self.steps >= cfg.WHOLESALE_STEPS_PER_TRIAL:
            done = True
            self.log_actions(self.demand_data[0], market_data[1], self.get_sum_purchased_for_ts(), real_action[1])
        return done

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
        self.active_target_timeslot = wd[0][0]

        self.demand_data: List[float] = list(dd)
        fc = self.get_forecast_for_active_ts()

    def apply_wholesale_averages(self, wd):
        avgs = calculate_running_averages(_get_wholesale_as_nparr(wd))
        for i, ts in enumerate([row[0] for row in wd]):
            self.wholesale_averages[ts] = avgs[i]

    def make_data_for_game(self, i):
        wholesale_file_path = self._wholesale_files[i]
        demand_file_path = self._demand_files[i]
        with open(wholesale_file_path) as file:
            wholesale_data = parse_wholesale_file(file)
        # let's reuse this
        # resetting first
        demand_data.clear()
        demand_data.parse_usage_game_log(demand_file_path)
        demand = demand_data.get_demand_data_values()
        idx = np.random.randint(0, high=len(demand), size=30)
        # using only random 30 picks from customers
        summed_random_30 = demand[idx, :].sum(axis=0)

        return trim_data(summed_random_30, wholesale_data,
                         demand_data.get_first_timestep_for_file(demand_file_path))

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
        # here is where the meat is.
        # first, amplify the action by 2 --> +1 == x2, -1 == x-2
        action = action * 2

        missing_energy = calculate_missing_energy(self.get_sum_purchased_for_ts(), self.demand_data[0])
        if self.latest_observeration is not None:
            latest_price = price_scaler.inverse_transform(np.array(self.latest_observeration[-1]).reshape(-1, 1))
        else:
            latest_price = 0
        amount = action[0] * missing_energy
        if amount > 0:
            # buying
            price = (-1) * latest_price * action[1]
        else:
            # selling
            price = latest_price * action[1]
        return np.array([amount, price])

    def get_market_data_now(self):
        """Returns the market data of the currently active timeslot and the current step
        """
        return np.array(self.wholesale_data[0][3:][self.steps])

    def get_sum_purchased_for_ts(self) -> float:
        if not self.purchases:
            return 0
        return np.array(self.purchases)[:, 0].sum()

    def get_forecast_for_active_ts(self):
        fc = self.demand_data[0]
        for i in range(cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL - self.steps):
            err_mult = random.uniform(-cfg.WHOLESALE_FORECAST_ERROR_PER_TS, cfg.WHOLESALE_FORECAST_ERROR_PER_TS)
            err_add = fc * err_mult
            fc += err_add
        return fc

    def make_observation(self) -> np.array:
        """Observation consists of : energy left to buy and last 24 known prices"""
        obs = []
        # adding required energy, and scaling
        required_energy = calculate_missing_energy(self.get_sum_purchased_for_ts(), self.demand_data[0])
        # scaling according to minmax math
        required_energy = demand_scaler.transform(np.array([required_energy]).reshape(-1, 1))
        obs.append(required_energy)

        # adding the prices after scaling
        prices = []
        for i in range(self.active_target_timeslot - 24, self.active_target_timeslot - self.steps):
            if i in self.wholesale_averages:
                prices.append(self.wholesale_averages[i])
            else:
                # it's a timestep from before the start of the game --> no idea what happened there with the logs files only
                prices.append(0)
        for i in range(self.steps):
            prices.append(self.wholesale_data[0][3 + i][1])
        prices = price_scaler.transform(np.array(prices).reshape(-1, 1)).flatten()
        obs.extend(prices)

        # turning it into an np array
        np_obs = np.array(obs)
        assert len(np_obs) == 25
        self.latest_observeration = np_obs
        return np_obs

    def calculate_squared_diff(self, bought):
        mse_diff = 0
        if self.demand_data[0] and bought:
            bought_sum = np.array(bought)[:, 0].sum()
            mse_diff = (bought_sum - self.demand_data[0]) ** 2
        return mse_diff

    def log_actions(self, demand_forecast, last_price, purchased_sum, price_offered):
        divergence = abs(purchased_sum / demand_forecast)
        price_diff = last_price - abs(price_offered)
        tb_writer_helper.write_any(divergence, 'divergence')
        tb_writer_helper.write_any(price_diff, 'price_diff')
