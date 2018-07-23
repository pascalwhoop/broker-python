import logging
import numpy as np

from agent_components.wholesale.environments.PowerTacEnv import PowerTacEnv
from agent_components.wholesale.util import calculate_energy_needed
from communication.grpc_messages_pb2 import PBOrderbook

log = logging.getLogger(__name__)

def get_action_translator(action_type):
    """based on the action type, the output of the NN means something different!"""
    if action_type == 'continuous':
        return continuous_translator
    if action_type == 'discrete':
        return discrete_translator
    if action_type == 'twoarmedbandit':
        return two_armed_bandit_translator
    #if we arrive here, it's not present
    raise NotImplementedError


def two_armed_bandit_translator(env: PowerTacEnv, actions):
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


def continuous_translator(env: PowerTacEnv, action):
    """similar to discrete translator but with a continuous action space"""
    assert env.predictions
    needed = calculate_energy_needed(env.predictions[-1], env.purchases)
    last_price = env.get_last_known_market_price()
    #positive if we sell, else negative
    base_bid = abs(last_price) if needed < 0 else - abs(last_price)
    base_action = np.array([needed, base_bid])
    real_action = action / 100 * base_action + base_action
    return real_action


def discrete_translator(env: PowerTacEnv, action):
    """This translator allows for several discrete actions in the price and amount dimensions. Each offer 10 actions
    Price: [-100% -- 20% steps -- +100%]   compared to last known price
    Amount: [-100% -- 20 % steps -- +100%] compared to prediction-purchased_already
    assume actions range from 0-9  map to -5 to +5 in 1 int steps
    """
    assert env.predictions
    needed = calculate_energy_needed(env.predictions[-1], env.purchases)
    last_price = env.get_last_known_market_price()

    action_perc = (action-5) * 20 / 100

    #positive if we sell, else negative
    base_bid = abs(last_price) if needed < 0 else - abs(last_price)

    base_action = np.array([needed, base_bid])
    real_action = action_perc * base_action + base_action
    return real_action