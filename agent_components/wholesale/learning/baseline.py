import numpy as np

from agent_components.wholesale.environments.PowerTacLogsMDPEnvironment import PowerTacLogsMDPEnvironment
from agent_components.wholesale.learning.reward_functions import simple_truth_ordering, shifting_balancing_price
from util.learning_utils import get_tb_cb, TbWriterHelper

model_name = "baseline-log-rl"
tag = ""
tb_writer = TbWriterHelper(model_name= model_name)

def get_instance(tag_, fresh):
    global tag
    tag = tag_
    return BaselineLearner()


class BaselineLearner:
    def __init__(self):
        self.env = PowerTacLogsMDPEnvironment(reward_func=shifting_balancing_price)
        self.nb_actions = 2
        self.env.new_game()
        self.memory_length = 1

        super().__init__()

    def learn(self):
        obs = self.env.reset()
        # always order what is missing and offer 10x the price. has to work
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
