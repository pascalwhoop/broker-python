from typing import List

from gym import Env

from agent_components.demand import data as demand_data
from agent_components.wholesale.environments.PowerTacEnv import PowerTacEnv
from communication.grpc_messages_pb2 import PBClearedTrade, PBMarketTransaction


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