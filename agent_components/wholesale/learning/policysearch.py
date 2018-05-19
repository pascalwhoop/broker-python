import multiprocessing
import sys

import tensorflow as tf
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import CnnPolicy, LnLstmPolicy, LstmPolicy, MlpPolicy

from agent_components.wholesale.mdp import PowerTacLogsMDPEnvironment


class PolicySearchLearner:
    def __init__(self, policy):
        self.policy = policy
        pass

    def train(self, num_timesteps):
        ncpu = multiprocessing.cpu_count()
        if sys.platform == 'darwin': ncpu //= 2
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=ncpu,
                                inter_op_parallelism_threads=ncpu)
        config.gpu_options.allow_growth = True  # pylint: disable=E1101
        tf.Session(config=config).__enter__()

        env = PowerTacLogsMDPEnvironment
        policy = {'cnn': CnnPolicy, 'lstm': LstmPolicy, 'lnlstm': LnLstmPolicy, 'mlp': MlpPolicy}[self.policy]
        ppo2.learn(policy=policy, env=env, nsteps=128, nminibatches=4,
                   lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
                   ent_coef=.01,
                   lr=lambda f: f * 2.5e-4,
                   cliprange=lambda f: f * 0.1,
                   total_timesteps=int(num_timesteps * 1.1))