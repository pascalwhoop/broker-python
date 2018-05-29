import ast
import csv
import logging
from typing import List

import numpy as np
from gym import spaces
from gym.spaces import Box
from sklearn import preprocessing

import util.config as cfg

# core component for the wholesale mdp process.

# long term: this should be able to both support decisions in an active competition as well as learn from an active
# competition and state files

# short term: just learning from state files, picking a competing broker from the state file that we wanna learn from


# should allow for several kinds of policy determining approaches --> DeepQ, PolicyGradient, ..

log = logging.getLogger(__name__)
sizes = np.finfo(np.array([1.0], dtype=np.float32)[0])
np_high = sizes.max
np_low = sizes.min

MIN_PRICE_SCALE = -200.0
MAX_PRICE_SCALE = 200.0
price_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
price_scaler.fit(np.array([MIN_PRICE_SCALE, MAX_PRICE_SCALE]).reshape(-1,1))

MIN_DEMAND = -100000.0
MAX_DEMAND = 100000.0
demand_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
demand_scaler.fit(np.array([MIN_DEMAND, MAX_DEMAND]).reshape(-1,1))

class WholesaleActionSpace(spaces.Box):
    """
    A wholesale action is anywhere in the 2D space with Dim 1 being limited to [-1,+1] and Dim 2 to [-2.0, +2.0]
        - Dimension 0 describes how much mWh to buy. Positive numbers buy, negative numbers sell The network gets fed a prediction regarding its portfolio balance.
          If the prediction is -243 (i.e. missing 243mWh for this timeslot to be matching the predicted demand) then
          - 0.5 means buying 243mWh.
          - 0 means buy nothing,
          - -0.5 means sell another 243mWh and
          - -1 means sell twice the predicted
          imbalance. If the prediction is +230 then 0.5 means buying another 230 and -0.5 means selling 230.
        - Dimension 1 describes the limit price. This is a mapping to the limit price.
          It's based on the known prices for the target timeslot.
             0 --> pay nothing
            -1 --> buy for 2x average known price
            +1 --> sell for 2x average known price
            +0.5 --> sell for average known price
            TODO is this maybe not large enough? increase box size for more freedom in pricing

    """

    def __init__(self):
        a = Box(low=np.array([-1.0, -1.0]), high=np.array([+1.0, +1.0]), dtype=np.float32)
        super().__init__(low=a.low, high=a.high, dtype=np.float32)


class WholesaleObservationSpace(spaces.Box):
    """
    - demand prediction - purchases 24x float
    - historical prices of currently traded TS 24x24 float (with diagonal TR-BL zeros in bottom right)
    - historical prices of last 168 timeslots
    - ... TODO more?
    """

    def __init__(self):
        # box needs min and max. using signed int32 min/max
       required_energy = Box(low=np_low, high=np_high, shape=(1,), dtype=np.float32)
       historical_prices = Box(low=np_low, high=np_high, shape=(168,), dtype=np.float32)
       current_prices = Box(low=np_low, high=np_high, shape=(24, 2), dtype=np.float32)
       super().__init__(low=np_low, high=np_high, shape=(1 + 24,), dtype=np.float32)


def _get_wholesale_as_nparr(wholesale_data: List):
    """Assumes it's being passed a list of wholesale data, where the first three columns are metadata and then it's raw stuff"""
    return np.array([row[3:] for row in wholesale_data])

def make_flat_observation(observation) -> np.array:
    obs = []
    obs.extend(observation['required_energy'])
    obs.extend(observation['historical_prices'])
    obs.extend(observation['current_prices'].flatten())
    return np.array(obs)


def unflat_action(action: np.array):
    return action.reshape(cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL, 2)


def get_do_nothing():
    return np.array([0,0])


def parse_wholesale_file(file):
    out = []
    reader = csv.reader(file)
    for row in reader:
        out.append([ast.literal_eval(str.strip(cell).replace(' ', ',')) for cell in row])
    return out
