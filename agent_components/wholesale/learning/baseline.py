import numpy as np
from rl.core import Agent

from agent_components.wholesale.environments.PowerTacEnv import PowerTacWholesaleAgent, PowerTacWholesaleObservation
from agent_components.wholesale.environments.PowerTacLogsMDPEnvironment import PowerTacLogsMDPEnvironment
from agent_components.wholesale.learning.reward_functions import simple_truth_ordering, shifting_balancing_price
from communication.grpc_messages_pb2 import PBOrderbook
from util.learning_utils import get_tb_cb, TbWriterHelper

model_name = "baseline-log-rl"
tag = ""
tb_writer = TbWriterHelper(model_name= model_name)

def get_instance(tag_, fresh):
    global tag
    tag = tag_
    return BaselineTrader()


class BaselineTrader(PowerTacWholesaleAgent):
    """This baseline trader does exactly what the prediction says. It takes the prediction, offers any price
    (10x the previous, so a lot) and tries to always balance the portfolio no matter what. It adapts the Keras-RL API
    not because I am using Keras here but because the other agents will too and this will talk to the same APIs and behave
    the same way as an NN based agents (except that it is just really stupid). """
    def forward(self, observation: PowerTacWholesaleObservation) -> np.array:
        """Takes the observation and returns the action that matches it"""

        #we need to buy the opposite of the customers predictions
        mWh = observation.predictions[-1] * -1
        if len(observation.orderbooks.values()) > 0:
            ob:PBOrderbook = list(observation.orderbooks.values())[-1]
            price = ob.clearingPrice * 10
        else:
            price = observation.hist_avg_prices[-1] * 10

        #if we buy mWh --> negative price
        if mWh > 0:
            price = abs(price) * -1
        #else positive price
        else:
            price = abs(price)

        return np.array([mWh, price])

    def backward(self, reward, terminal):
        """Does nothing really. """
        pass


    def __init__(self):
        self.env = PowerTacLogsMDPEnvironment(reward_func=shifting_balancing_price)
        self.nb_actions = 2
        self.env.new_game()
        self.memory_length = 1

        super().__init__()

    def learn(self):
        obs = self.env.reset()
        # always order what is missing and offer 10x the price. will almost always manage to balance the portfolio and that
        # early in the bidding chain
        action = np.array([0.5, 5])
        for i in range(100000):
            done = False
            reward_episode = 0
            while not done:
                obs, reward, done, info = self.env.step(action)
                reward_episode += reward
            # timeslot completed, reset
            obs = self.env.reset()
            tb_writer.write_any(reward_episode, "episode_reward")







    def make_logger_callback(self):
        return get_tb_cb("{}_{}".format(model_name, tag), write_graph=False)
