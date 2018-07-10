import datetime
import json
import logging
import numpy as np
import os
from  util.learning_utils import TbWriterHelper 
from tensorforce.agents import Agent

import util.config as cfg
from agent_components.wholesale.environments.PowerTacEnv import PowerTacEnv, PowerTacWholesaleAgent, \
    PowerTacWholesaleObservation
from agent_components.wholesale.learning.reward_functions import market_relative_prices, step_close_to_prediction_reward
from communication.grpc_messages_pb2 import PBOrderbook

log = logging.getLogger(__name__)

MODEL_NAME = "cdq_v2_inverse_agent" + str(datetime.datetime.now())
tag = ""
BATCH_SIZE = 32
# TODO determine input shape
NN_INPUT_SHAPE = (
    cfg.WHOLESALE_HISTORICAL_DATA_LENGTH + cfg.DEMAND_FORECAST_DISTANCE + cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL,)
nb_actions = 2


def load_spec(tag):
    p = os.path.join(cfg.WHOLESALE_TENSORFORCE_CONFIGS, "{}.json".format(tag))
    with open(p) as f:
        return json.load(f)


def get_instance(tag_, fresh):
    global tag
    tag = tag_
    spec = load_spec(tag)
    kwargs = model_kwargs[tag]
    return TensorforceAgent(spec, kwargs)


class TensorforceAgent(PowerTacWholesaleAgent):
    """
    A "environment centric" agent. This agent doesn't control the flow of processing but instead simply offers `forward` and `backward` APIs
    which return actions for observations and learn from past actions and rewards.
    """

    def __init__(self, spec, kwargs):
        self.agent = Agent.from_spec(spec, kwargs=kwargs)
        #TODO should be in the PowerTacWholesaleAgent as an inherited thing for all agents
        self.tb_log_helper = TbWriterHelper("dqn", True)

    def forward(self, env: PowerTacEnv):
        obs = self.make_observation(env)
        env.observations.append(obs)
        nn_action, states, internals = self.agent.act(obs, buffered=False)
        actions = translate_two_armed(env, nn_action)
        return actions, nn_action, internals

    def make_observation(self, env: PowerTacEnv):
        purchases = np.array([p.mWh for p in env.purchases])
        hist_prices = np.array(env._historical_prices)
        predictions = env.predictions
        # padding properly to keep same position and size
        pad = 24 - len(purchases)
        purchases = np.pad(purchases, (0, pad), 'constant', constant_values=0)
        pad = 168 - len(hist_prices)
        hist_prices = np.pad(hist_prices, (0, pad), 'constant', constant_values=0)
        pad = 24 - len(predictions)
        predictions = np.pad(predictions, (0, pad), 'constant', constant_values=0)
        obs = np.concatenate((predictions, hist_prices, purchases))
        return obs

    def backward(self, env: PowerTacEnv, action, reward):
        obs = env.observations[-1]
        #action = env.actions[-1]
        action = env.nn_actions[-1]
        terminal = env._step > 23
        last_purchase = env.purchases[-1].mWh if env.purchases else 0
        self.tb_log_helper.write_any(reward, "reward")

        try:
            return self.agent.atomic_observe(obs, action, env.internals[-1], reward, terminal)
        except Exception as e:
            log.exception(e)



# =========================== Agent configs

model_configs = {
    'base': dict(
        states={'shape': NN_INPUT_SHAPE, 'type': "float"},
        actions={'shape': (2,), 'type': 'float'},

    ),
    # 
    'discrete': dict(
        states={'shape': NN_INPUT_SHAPE, 'type': "float"},
        actions={'shape': (2,), 'type': 'int'}
    ),

    'twoarmedbandit': dict(
        states= {'shape': NN_INPUT_SHAPE, 'type': "float"},
        actions={'shape':(1,),            'type':'int', 'num_actions': 2}
    )
}

model_kwargs = {
    'random': {**model_configs['base']},
    'naf': {**model_configs['base'], **{'network': load_spec('mlp2_network')}}, # same as trpo
    'trpo': {**model_configs['base'], **{'network': load_spec('mlp2_network')}},# suffers from some bug that lets it crash on start
    'vpg': {**model_configs['base'], **{'network': load_spec('mlp2_network')}}, # not yet tested
    'ppo': {**model_configs['base'], **{'network': load_spec('mlp2_network')}}, # ppo bug https://github.com/reinforceio/tensorforce/issues/391
    'dqn': {**model_configs['base'], **{'network': load_spec('mlp2_normalized_network')}}
    #'dqn': {**model_configs['twoarmedbandit'], **{'network': load_spec('mlp2_normalized_network')}}
}


def translate_two_armed(env: PowerTacEnv, actions):
    if not actions[0]:
        log.warning("agent chose random stupid action, shouldn't")
        return np.random.randint(-100,100, size=2)
    # we need to buy the opposite of the customers predictions
    mWh = env.predictions[-1]
    # but reduce it by what we already purchased
    bought = np.array([a.mWh for a in env.purchases]).sum()
    mWh = mWh + bought
    mWh *= -1

    if len(env.orderbooks) > 0:
        ob: PBOrderbook = env.orderbooks[-1]
        price = ob.clearingPrice * 10
    else:
        price = env._historical_prices[-24]
    # if we buy mWh --> negative price
    if mWh > 0:
        price = abs(price) * -1 * 10  # offering 10 x the price
    # else positive price
    else:
        price = abs(price) / 10  # asking 1/10th the market price

    return [mWh, price]

