import datetime

from keras import Model, Sequential
from keras.callbacks import TerminateOnNaN
from keras.layers import Activation, Concatenate, Dense, Flatten, Input
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from agent_components.wholesale.mdp import PowerTacLogsMDPEnvironment
from util.learning_utils import get_tb_cb

model_name = "continuous-deepq" + str(datetime.datetime.now())
tag = ""


def get_instance(tag_, fresh):
    global tag
    tag = tag_
    return ContinuousDeepQLearner()


class ContinuousDeepQLearner:
    def __init__(self):
        self.env = PowerTacLogsMDPEnvironment()
        self.nb_actions = 2
        self.env.new_game()
        self.memory_length = 1

        super().__init__()

    def is_done(self):
        return False

    def learn(self):
        action_input = Input(shape=(self.nb_actions,), name='action_input')
        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=100000, window_length=self.memory_length)
        random_process = OrnsteinUhlenbeckProcess(size=self.nb_actions, theta=.2, mu=0., sigma=.3)
        agent = DDPGAgent(nb_actions=self.nb_actions,               #number of actions
                          actor=self.make_actor(),                  #actor --> polcy value
                          critic=self.make_critic(action_input),    #q value estimator
                          critic_action_input=action_input,         #tensorflow wrapper for input shape
                          memory=memory,                            #how long is the memory of the agent?
                          nb_steps_warmup_critic=100,               #
                          nb_steps_warmup_actor=100,                #
                          random_process=random_process,            #causes exploration
                          gamma=1, target_model_update=1e-3)        #
        agent.compile(Adam(lr=.000001, clipnorm=1.), metrics=['mae'])#

        # Okay, now it's time to learn something! We visualize the training here for show, but this
        # slows down training quite a lot. You can always safely abort the training prematurely using
        # Ctrl + C.
        agent.fit(self.env, nb_steps=50000000, log_interval=1000, visualize=False, verbose=1,
                  callbacks=[self.make_logger_callback(), TerminateOnNaN()])

        # After training is done, we save the final weights.
        agent.save_weights('ddpg_{}_weights.h5f'.format('offline'), overwrite=True)

        # Finally, evaluate our algorithm for 5 episodes.
        agent.test(self.env, nb_episodes=5, visualize=False, nb_max_episode_steps=200)

    def make_actor(self):
        # Next, we build a very simple model.
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + self.env.observation_space.shape))
        actor.add(Dense(self.env.observation_space.shape[0], bias_initializer='zeros'))
        actor.add(Activation('linear'))
        actor.add(Dense(128, bias_initializer='zeros'))
        actor.add(Activation('linear'))
        actor.add(Dense(128, bias_initializer='zeros'))
        actor.add(Activation('linear'))
        actor.add(Dense(self.nb_actions, bias_initializer='zeros'))
        actor.add(Activation('linear'))
        #        actor.add(Activation('relu'))
        #        actor.add(Dense(128, bias_initializer='zeros'))
        #        actor.add(Activation('relu'))
        #        actor.add(Dense(self.nb_actions, bias_initializer='zeros'))
        #actor.add(Activation('tanh'))
        print(actor.summary())
        return actor

    def make_critic(self, action_input):
        observation_input = Input(shape=(1,) + self.env.observation_space.shape, name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = Concatenate()([action_input, flattened_observation])
        x = Dense(50, bias_initializer='zeros')(x)
        x = Activation('linear')(x)
        x = Dense(50, bias_initializer='zeros')(x)
        x = Activation('linear')(x)
        x = Dense(50, bias_initializer='zeros')(x)
        x = Activation('linear')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        #        x = Concatenate()([action_input, flattened_observation])
        #        x = Dense(128, bias_initializer='zeros')(x)
        #        x = Activation('relu')(x)
        #        x = Dense(128, bias_initializer='zeros')(x)
        #        x = Activation('relu')(x)
        #        x = Dense(128, bias_initializer='zeros')(x)
        #        x = Activation('relu')(x)
        #        x = Dense(1)(x)
        #        x = Activation('linear')(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)
        print(critic.summary())
        return critic

    def make_logger_callback(self):
        return get_tb_cb("{}_{}".format(model_name, tag), write_graph=False)
