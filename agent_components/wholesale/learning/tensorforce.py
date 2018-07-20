"""The second attempt at getting a RL agent up and running within Python when talking to PowerTAC. This agent makes use of the TensorForce library (soon to be replaced with YARL).
 It's still in a very rough form, doesn't support loading/saving of the model yet etc. """
import datetime
import json
import logging
import numpy as np
import os

from agent_components.wholesale.learning import reward_functions
from agent_components.wholesale.learning.postprocessor import get_action_translator
from tensorforce.agents import Agent

import util.config as cfg
from agent_components.wholesale.environments.PowerTacEnv import PowerTacEnv
from agent_components.wholesale.environments.PowerTacWholesaleAgent import PowerTacWholesaleAgent
from agent_components.wholesale.learning.preprocessor import get_observation_preprocessor

log = logging.getLogger(__name__)

MODEL_NAME = "cdq_v2_inverse_agent" + str(datetime.datetime.now())
BATCH_SIZE = 32
NN_INPUT_SHAPE = ( cfg.WHOLESALE_HISTORICAL_DATA_LENGTH + cfg.DEMAND_FORECAST_DISTANCE + cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL,)

model_configs = {
    'continuous': dict(
        states={'shape': NN_INPUT_SHAPE, 'type': "float"},
        actions={'shape': (2,), 'type': 'float'},

    ),
    #
    'discrete': dict(
        states={'shape': NN_INPUT_SHAPE, 'type': "float"},
        actions={'shape': (2,), 'type': 'int', 'num_actions': 11}
    ),

    'twoarmedbandit': dict(
        states= {'shape': NN_INPUT_SHAPE, 'type': "float"},
        actions={'shape':(1,),            'type':'int', 'num_actions': 2}
    )
}

def load_spec_file(spec):
    p = os.path.join(cfg.WHOLESALE_TENSORFORCE_CONFIGS, "{}.json".format(spec))
    with open(p) as f:
        return json.load(f)


def create_spec(action_type, agent_type, network):
    """Combines all the information to an tensorforce agent spec"""
    agent_spec = load_spec_file(agent_type)
    network = load_spec_file(network)
    agent_spec['network'] = network
    agent_spec = {**agent_spec, **model_configs[action_type]}
    return agent_spec


class TensorforceAgent(PowerTacWholesaleAgent):
    """
    A "environment centric" agent. This agent doesn't control the flow of processing but instead simply offers `forward` and `backward` APIs
    which return actions for observations and learn from past actions and rewards.
    """

    def __init__(self, agent_type, network, action_type, preprocessor_type,reward, tag):
        rf = reward_functions.__dict__[reward]
        super().__init__("-".join([agent_type, network, action_type, reward, tag]))
        agent_spec = create_spec(action_type, agent_type, network)
        self._tf_agent = Agent.from_spec(agent_spec, {})
        self.action_translator = get_action_translator(action_type)
        self.preprocessor = get_observation_preprocessor(preprocessor_type)

    def forward(self, env: PowerTacEnv):
        obs = self.preprocessor(env)
        env.observations.append(obs)
        nn_action, states, internals = self._tf_agent.act(obs, buffered=False)
        action = self.action_translator(env, nn_action)
        action[1] = env.predictions[-1]*10
        return action, nn_action, internals


    def backward(self, env: PowerTacEnv, action, reward):
        obs = env.observations[-1]
        #action = env.actions[-1]
        action = env.nn_actions[-1]
        terminal = env._step > 23
        last_purchase = env.purchases[-1].mWh if env.purchases else 0
        self.tb_log_helper.write_any(reward, "reward")

        try:
            return self._tf_agent.atomic_observe(obs, action, env.internals[-1], reward, terminal)
        except Exception as e:
            log.exception(e)


