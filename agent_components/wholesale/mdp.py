import ast
import csv
import random
from collections import deque
from typing import List

import numpy as np
from gym import spaces
from gym.core import Env
from gym.spaces import Box

from agent_components.demand.data import DemandForecasts, parse_usage_game_log, get_demand_data_values
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
        - Dimension 0 describes how much kWh to buy. The network gets fed a prediction regarding its portfolio balance.
          If the prediction is -243 (i.e. missing 243kWh for this timeslot to be matching the predicted demand) then 0.5
          means buying 243kWh. 0 means buy nothing, -0.5 means sell another 243kWh and -1 means sell twice the predicted
          imbalance. If the prediction is +230 then 0.5 means buying another 230 and -0.5 means selling 230.
        - Dimension 1 describes the limit price. This is a mapping to the limit price. It's based on TODO (probably historical average or something)
    """

    def __init__(self):
        timestep_actions = []
        for i in range(cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL):
            a = Box(low=np.array([-1.0, -2.0]), high=[+1.0, +2.0])
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
        self.forecasts: List[DemandForecasts]

    def update_forecasts(self, fc: DemandForecasts):
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

        self._prices_files = get_wholesale_file_paths()
        self._usages_files = get_usage_file_paths()
        self.initial_timeslot = 0
        self.active_timeslots = deque(maxlen=24)


    def step(self, action: WholesaleActionSpace):
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

        game_numbers = self._make_random_game_order()
        # iterate over past games randomly
        for i in game_numbers:
            self.make_data_for_game(i)





        # evaluate
        # block until answers are all collected
        # return with observation (TODO build this), reward (TODO calc) and done if reached terminal state

        pass

    def make_data_for_game(self, i):
        price_file = self._prices_files[i]
        usage_file = self._usages_files[i]
        with open(price_file) as file:
            wholesale_data = self.parse_wholesale_file(file)
        # let's reuse this
        parse_usage_game_log(usage_file)
        demand = get_demand_data_values()
        summed_first_30 = demand[:30,:].sum(axis=0)
        return wholesale_data, summed_first_30

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

    def reset(self) -> None:
        """Marks the environment as completed and therefore lets the agent learn again on a new game once it is ready"""
        self.active_timeslots.clear()
        pass

    def render(self, mode='logging') -> None:
        """
        Nothing to render. Although  may log or output some stuff to tensorboard. TBD
        """
        pass

    def close(self) -> None:
        pass

    def parse_wholesale_file(self, file):
        out = []
        reader = csv.reader(file)
        for row in reader:
            out.append([ast.literal_eval(str.strip(cell).replace(' ', ',')) for cell in row])
        return out

