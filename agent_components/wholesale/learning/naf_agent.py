import datetime

import numpy as np
from keras import Model, Sequential
from keras.callbacks import TerminateOnNaN
from keras.layers import Activation, Concatenate, Dense, Flatten, Input, CuDNNGRU
from keras.optimizers import Adam
from rl.agents import DDPGAgent, NAFAgent
from rl.memory import SequentialMemory
from rl.processors import WhiteningNormalizerProcessor
from rl.random import OrnsteinUhlenbeckProcess
from rl.util import WhiteningNormalizer

from agent_components.wholesale.environments.PowerTacLogsMDPEnvironment import PowerTacLogsMDPEnvironment
from agent_components.wholesale.learning.reward_functions import direct_cash_reward, simple_truth_ordering
from agent_components.wholesale.mdp import WholesaleObservationSpace
from util.learning_utils import get_tb_cb

MODEL_NAME = "continuous-deepq-naf" + str(datetime.datetime.now())
tag = ""


BATCH_SIZE=32
NN_INPUT_SHAPE = (1,) + WholesaleObservationSpace().shape

env = PowerTacLogsMDPEnvironment(reward_func=direct_cash_reward)


def get_instance(tag_, fresh):
    global tag
    tag = tag_
    return ContinuousDeepQLearner()


class ContinuousDeepQLearner:
    def __init__(self):
        self.nb_actions = 2
        env.new_game()
        self.memory_length = 1

        super().__init__()

    def is_done(self):
        return False

    def learn(self):
        nb_actions = self.nb_actions

        # Build all necessary models: V, mu, and L networks.
        V_model = Sequential()
        V_model.add(Flatten(input_shape=NN_INPUT_SHAPE))
        V_model.add(Dense(16))
        V_model.add(Activation('linear'))
        V_model.add(Dense(16))
        V_model.add(Activation('linear'))
        V_model.add(Dense(16))
        V_model.add(Activation('linear'))
        V_model.add(Dense(1))
        V_model.add(Activation('linear'))
        print(V_model.summary())

        mu_model = Sequential()
        mu_model.add(Flatten(input_shape=NN_INPUT_SHAPE))
        mu_model.add(Dense(16))
        mu_model.add(Activation('relu'))
        mu_model.add(Dense(16))
        mu_model.add(Activation('relu'))
        mu_model.add(Dense(16))
        mu_model.add(Activation('relu'))
        mu_model.add(Dense(nb_actions))
        mu_model.add(Activation('linear'))
        print(mu_model.summary())

        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=NN_INPUT_SHAPE, name='observation_input')
        x = Concatenate()([action_input, Flatten()(observation_input)])
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(((nb_actions * nb_actions + nb_actions) // 2))(x)
        x = Activation('linear')(x)
        L_model = Model(inputs=[action_input, observation_input], outputs=x)
        print(L_model.summary())

        action_input = Input(shape=(self.nb_actions,), name='action_input')
        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=100000, window_length=self.memory_length)
        random_process = OrnsteinUhlenbeckProcess(size=self.nb_actions, theta=.002, mu=0., sigma=.3)
        #creating a normalizing processor. The agent internally normalizes over our already scaled input.
        normalizing_processor = WhiteningNormalizerProcessor()
        agent = NAFAgent(nb_actions=self.nb_actions,  #number of actions
                         V_model=V_model,  #policy value
                         batch_size=BATCH_SIZE,
                         L_model=L_model,
                         mu_model=mu_model,
                         memory=memory,  #how long is the memory of the agent?
                         nb_steps_warmup=100,  #
                         random_process=random_process,  #causes exploration
                         processor=normalizing_processor,
                         gamma=1, target_model_update=1e-3)        #
        agent.compile(Adam(lr=.000001, ), metrics=['mae'])#

        # Okay, now it's time to learn something! We visualize the training here for show, but this
        # slows down training quite a lot. You can always safely abort the training prematurely using
        # Ctrl + C.
        logger_cb = self.make_logger_callback()
        agent.fit(env, nb_steps=50000000, log_interval=1000, visualize=False, verbose=1,
                  callbacks=[logger_cb, TerminateOnNaN()])

        # After training is done, we save the final weights.
        agent.save_weights('ddpg_{}_weights.h5f'.format('offline'), overwrite=True)

        # Finally, evaluate our algorithm for 5 episodes.
        agent.test(env, nb_episodes=5, visualize=False, nb_max_episode_steps=200)




    def make_logger_callback(self):
        return get_tb_cb("{}_{}".format(MODEL_NAME, tag), write_graph=False)
