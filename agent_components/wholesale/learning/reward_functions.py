import math
import util.config as cfg

import numpy as np

from agent_components.wholesale.environments.PowerTacEnv import PowerTacEnv
from agent_components.wholesale.util import calculate_running_averages, calculate_du_fee, average_price_for_power_paid, \
    calculate_balancing_needed


# what can rewards be based on?
# - action
# - realized cost
# - divergence from forecast
from util.utils import deprecated


@deprecated
def simple_truth_ordering(env, action, market_trades, purchases, realized_usage):
    """
    This helper function trains the agent to initially always order exactly the amount it has forecasted to need and
    offer a price that is always 10% better than the market price --> always clearing
    :param action:
    :return:
    """
    amount = action[0]
    price = action[1]
    return -((amount-0.50)**2 + (price-0.60)**2)

def market_relative_prices(env:PowerTacEnv):
    market_trades = [[tr.executionMWh, tr.executionPrice] for tr in env.cleared_trades]
    purchases = [[p.mWh, p.price] for p in env.purchases]
    realized_usage = env.realized_usage

    average_market = calculate_running_averages(np.array([market_trades]))[0]

    balancing_needed = calculate_balancing_needed(purchases, realized_usage)

    # TODO for now just a fixed punishment for every balanced mWh. Later maybe based on balancing stats data
    du_trans = calculate_du_fee(average_market, balancing_needed)
    if du_trans:
        purchases.append(du_trans)

    type_, average_agent = average_price_for_power_paid(purchases)
    market_relative_prices = 0
    if type_ == 'ask':
        # broker is overall selling --> higher is better
        market_relative_prices = average_agent / average_market
    if type_ == 'bid':
        # broker is overall buyer --> lower is better
        market_relative_prices = average_market / average_agent
    #large market_relative_prices --> good
    return -100 + (market_relative_prices* 100)


@deprecated
def direct_cash_reward(env, action, market_trades, purchases, realized_usage):
    """Gives back the direct relation between paid amount and what the agent would have paid if it achieved average costs"""
    average_market = calculate_running_averages(np.array([market_trades]))[0]

    balancing_needed = calculate_balancing_needed(purchases, realized_usage)

    # TODO for now just a fixed punishment for every balanced mWh. Later maybe based on balancing stats data
    du_trans = calculate_du_fee(average_market, balancing_needed)
    if du_trans:
        purchases.append(du_trans)

    # summing the entire payments. Because one of both is usually negative, abs of mWh amount
    # negative value --> overall casflow negative -->
    sum_agent = np.array([abs(r[0])*r[1] for r in purchases]).sum()
    # average_market is always positive --> negative mWh amount --> reducing costs --> fair, because "what if sold"
    sum_agent_with_average_prices = realized_usage * abs(average_market)

    #logging to tensorboard
    p = np.array(purchases)

    #TODO both do the same? check again
    if realized_usage > 0:
        # we are selling energy! it's good if we're positive here!
        # if agent positive --> if sold better than average -> above 0
        #                   --> if sold worse than  average -> below 0
        # if agent negative --> giving difference between the two
        return sum_agent - sum_agent_with_average_prices
    if realized_usage <= 0:
        # we are buying energy --> negative
        # if agent positive --> crazy good
        # if agent negative --> if smaller than sum_avg --> above 0
        #                   --> if larger than  sum_avg --> below 0
        return sum_agent - sum_agent_with_average_prices




def step_close_to_prediction_reward(env: PowerTacEnv):
    """A reward function that rewards actions that order energy close to what was forecasted. This allows the agent
    to deviate from the forecasts a little but not too much. Generally, the forecast is supposed to be considered as
     a "true" value, i.e. it's assumed that the predictor knows better than the wholesale trader. """
    # motivates to ensure the portfolio is covered the closer we get to the final timestep.
    latest_news = env.predictions[-1]
    purchased_already = np.array([p.mWh for p in env.purchases]).sum()
    needed = latest_news + purchased_already
    action = env.actions[-1]
    return -abs((action[0] - needed)) * (env._step/ cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL)



def shifting_balancing_price(env:PowerTacEnv):
    """Gives back a relation between the average market price for the target timeslot and the average price the broker achieved"""
    market_trades = [[tr.executionMWh, tr.executionPrice] for tr in env.cleared_trades]
    purchases = [[p.mWh, p.price] for p in env.purchases]
    realized_usage = env.realized_usage


    average_market = calculate_running_averages(np.array([market_trades]))[0]

    balancing_needed = calculate_balancing_needed(purchases, realized_usage)

    # TODO for now just a fixed punishment for every balanced mWh. Later maybe based on balancing stats data
    du_trans = calculate_du_fee(average_market, balancing_needed)
    if du_trans:
        purchases.append(du_trans)

    type_, average_agent = average_price_for_power_paid(purchases)
    market_relative_prices = 0
    if type_ == 'ask':
        # broker is overall selling --> higher is better
        market_relative_prices = average_agent / average_market
    if type_ == 'bid':
        #broker is overall buyer --> lower is better
        market_relative_prices = average_market / average_agent

    #reward is made up of several components
    # priority one: getting the mWh that are predicted
    # priority number two: getting those cheaply

    #if balancing high --> ratio close to 1
    bn_abs = abs(balancing_needed)
    weight_balancing = bn_abs / (abs(realized_usage) + bn_abs)
    weight_price = 1 - weight_balancing
    # the two components of the reward function
    balancing_reward_component = weight_balancing * (1 / balancing_needed ** 2)
    price_reward_component = weight_price * market_relative_prices
    # making sure we don't have nan's or inf's
    balancing_reward_component = balancing_reward_component if not math.isnan(balancing_reward_component) else 0
    price_reward_component = price_reward_component if not math.isnan(price_reward_component) else 0
    reward = balancing_reward_component + price_reward_component
    return reward
    #rdreturn market_relative_prices - (balancing_needed/ (balancing_needed + energy_sum))
