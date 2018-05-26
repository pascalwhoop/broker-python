from keras import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from util.learning_utils import get_callbacks, get_tb_cb
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from agent_components.wholesale.mdp import PowerTacLogsMDPEnvironment

model_name = "continuous-deepq"
tag = ""

def get_instance(tag_, fresh):
    global tag
    tag = tag_
    return ContinuousDeepQLearner()

class ContinuousDeepQLearner:
    def __init__(self):
        self.env = PowerTacLogsMDPEnvironment()
        self.nb_actions = 48
        self.env.new_game()

        super().__init__()

    def is_done(self):
        return False

    def learn(self):
        action_input = Input(shape=(self.nb_actions,), name='action_input')
        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=100000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(size=self.nb_actions, theta=.15, mu=0., sigma=.3)
        agent = DDPGAgent(nb_actions=self.nb_actions, actor=self.make_actor(), critic=self.make_critic(action_input),critic_action_input=action_input,
        memory = memory, nb_steps_warmup_critic = 100, nb_steps_warmup_actor = 100,
        random_process = random_process, gamma = .99, target_model_update = 1e-3)
        agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

        # Okay, now it's time to learn something! We visualize the training here for show, but this
        # slows down training quite a lot. You can always safely abort the training prematurely using
        # Ctrl + C.
        agent.fit(self.env, nb_steps=50000, log_interval=100, visualize=False, verbose=1, nb_max_episode_steps=200, callbacks=[self.make_logger_callback()])

        # After training is done, we save the final weights.
        agent.save_weights('ddpg_{}_weights.h5f'.format('offline'), overwrite=True)

        # Finally, evaluate our algorithm for 5 episodes.
        agent.test(self.env, nb_episodes=5, visualize=False, nb_max_episode_steps=200)


    def make_actor(self):
        # Next, we build a very simple model.
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + self.env.observation_space.shape))
        actor.add(Dense(128))
        actor.add(Activation('relu'))
        actor.add(Dense(128))
        actor.add(Activation('relu'))
        actor.add(Dense(128))
        actor.add(Activation('relu'))
        actor.add(Dense(self.nb_actions))
        actor.add(Activation('linear'))
        print(actor.summary())
        return actor

    def make_critic(self, action_input):
        observation_input = Input(shape=(1,) + self.env.observation_space.shape, name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = Concatenate()([action_input, flattened_observation])
        x = Dense(128)(x)
        x = Activation('relu')(x)
        x = Dense(128)(x)
        x = Activation('relu')(x)
        x = Dense(128)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)
        print(critic.summary())
        return critic

    def make_logger_callback(self):
        return get_tb_cb("{}_{}".format(model_name, tag), write_graph=False)


