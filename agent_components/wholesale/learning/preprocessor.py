import numpy as np

from agent_components.wholesale.environments.PowerTacEnv import PowerTacEnv
from agent_components.wholesale.util import price_scaler, demand_scaler


def get_observation_preprocessor(preprocessor_type):
    if preprocessor_type == "simple":
        return simple_hist_and_preds
    if preprocessor_type == "simplenorm":
        return norm_hist_and_preds
    #if arrived here, function doesn't exist
    raise NotImplementedError


def simple_hist_and_preds(env: PowerTacEnv) -> np.array:
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

def norm_hist_and_preds(env: PowerTacEnv):
    """the same as simple_hist_and_preds but with a normalized input scaling"""
    data = simple_hist_and_preds(env)
    data[0:24] = demand_scaler.transform(data[0:24].reshape(-1,1)).flatten()
    data[24:24+168] = price_scaler.transform(data[24:24+168].reshape(-1,1)).flatten()
    data[24+168:] = demand_scaler.transform(data[24+168:].reshape(-1,1)).flatten()
    return data
