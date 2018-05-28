from keras import Model, Sequential
from keras.callbacks import TerminateOnNaN
from keras.layers import Activation, Concatenate, Dense, Flatten, Input
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from agent_components.wholesale.mdp import PowerTacLogsMDPEnvironment
from util.learning_utils import get_tb_cb

model_name = "continuous-deepq"
tag = ""


def get_instance(tag_, fresh):
    global tag
    tag = tag_
    return BaselineLearner()


class BaselineLearner:
    def __init__(self):
        self.env = PowerTacLogsMDPEnvironment()
        self.nb_actions = 2
        self.env.new_game()
        self.memory_length = 1

        super().__init__()

    def learn(self):
        pass


    def make_logger_callback(self):
        return get_tb_cb("{}_{}".format(model_name, tag), write_graph=False)
