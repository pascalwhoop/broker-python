import math

import logging

import util.config as cfg

import numpy as np

from agent_components.wholesale.environments.PowerTacEnv import PowerTacEnv
from agent_components.wholesale.util import calculate_running_averages, calculate_du_fee, average_price_for_power_paid, \
    calculate_balancing_needed, calculate_energy_needed, is_cleared_with_volume_probability

# what can rewards be based on?
# - action
# - realized cost
# - divergence from forecast
from communication.grpc_messages_pb2 import PBOrder
from util.utils import deprecated


log = logging.getLogger(__name__)
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
    """Calculates the relationship between the agents prices paid and the average market price at the end of the sessi on"""
    #this only allows the agent to receive feedback at the end of the timeslot, so at the last trading opportunity.
    #if env._step != 25:
    #    return 0
    market_trades = [[tr.executionMWh, tr.executionPrice] for tr in env.cleared_trades]
    #all purchases as list
    purchases = [[p.mWh, p.price] for p in env.purchases]
    #adding balancing TX
    #TEST removing the balancing charge --> only dependent on what the broker actually bought
    #purchases.append([env.balancing_tx.kWh / 1000 * -1, env.balancing_tx.charge])

    #getting the averages for market and broker purchases
    average_market = calculate_running_averages(np.array([market_trades]))[0]
    type_, average_agent = average_price_for_power_paid(purchases)

    mrp = 0
    if type_ == 'ask' and average_market != 0:
        # broker is overall selling --> higher is better
        mrp = average_agent / average_market
    if type_ == 'bid' and average_agent != 0:
        # broker is overall buyer --> lower is better
        mrp = average_market / average_agent
    return mrp


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


#def reduce_imbalance_cheap_at_end(env:PowerTacEnv):
#    """every time there is still energy left to be purchased, a negative reward is given. additionally, the market_relative_prices areused with a factor of 100"""
#    latest_news = env.predictions[-1] if env.predictions else 0
#    # sum purchased
#    needed = calculate_energy_needed(latest_news, env.purchases)
    # PROBLEM: -needed gives negative reward in round 0 but the agent isn't at fault, it had no chance to buy yet
#    return -needed + 100 * market_relativeis_prices(env)


# PROBLEM bad structure... converges towards attempting to buy nothing because that always has a likelihood of 1
#def succ_orders_then_cheap(env:PowerTacEnv):
#    """gives back the probability of the action clearing. this makes the agent learn to always bid so that the market
#    clears """
#    if not env.actions or not env.cleared_trades:
#        #TODO 0 or something else? is the reward neg or
#        return 0
#    mWh = env.actions[-1][0]
#    price = env.actions[-1][0]
#    o = PBOrder(mWh=mWh, limitPrice=price)
#    last_ct = env.cleared_trades[-1]
#    cl, r_prob = is_cleared_with_volume_probability(o, np.array([last_ct.executionMWh, last_ct.executionPrice]))
#    #trial 1... just the probability of clearing is learned
#    return r_prob

def balancing_reward(env: PowerTacEnv):
    """punishes balancing required by the DU to encourage good purchasing ahead of time"""
    balanced_mWh = env.balancing_tx.kWh / 1000 * -1
    overall_consume = env.realized_usage
    part_balanced = abs(balanced_mWh / overall_consume)
    env.agent.tb_log_helper.write_any(part_balanced, "part_balanced")
    return -part_balanced


def only_final_step(env:PowerTacEnv):
     if env._step >= 25:
        # usually positive, the larger the better
        r_rel = market_relative_prices(env)
        #usually negative, the closer to 0 the better
        r_balancing = balancing_reward(env)
        part_balancing = - r_balancing
        #final step
        #if lots of balancing --> r_balancing is important
        #if little balancing --> relative price is important
        return r_rel * (1-part_balancing) + r_balancing * (part_balancing)
     else:
        return 0


def step_close_relative_mprice(env: PowerTacEnv):
    """2 part reward function:
        in the first 24 timesteps, it encourages to step close to the missing energy
        in the last step, it encourages to have little balancing and gives feedback on the average price paid for the TS
    """
    if env._step >= 25:
        # usually positive, the larger the better
        r_rel = market_relative_prices(env)
        #usually negative, the closer to 0 the better
        r_balancing = balancing_reward(env)
        #final step
        return r_rel + r_balancing
    else:
        #usually negative, the closer to 0 the better
        return step_close_to_prediction_reward(env)

@deprecated
def unified_step_close_relative_market_rel_mprice(env:PowerTacEnv):
    #TODO add factor \alpha for weighing the components
    #usually negative, the closer to 0 the better
    r_pred = step_close_to_prediction_reward(env)
    # usually positive, the larger the better
    r_rel = market_relative_prices(env) if env._step == 25 else 0
    r_price = close_to_market_price(env)
    log.info("r_pred {} r_rel {} r_price {}".format(r_pred, r_rel, r_price))
    return r_pred + r_rel + r_price

def close_to_market_price(env:PowerTacEnv):
    r_price = 0
    if not env.cleared_trades or not env.actions:
        return r_price
    return -abs(env.cleared_trades[-1].executionPrice - env.actions[-1][1])


def step_close_to_prediction_reward(env: PowerTacEnv):
    """A reward function that rewards actions that order energy close to what was forecasted. This allows the agent
    to deviate from the forecasts a little but not too much. Generally, the forecast is supposed to be considered as
     a "true" value, i.e. it's assumed that the predictor knows better than the wholesale trader. """
    # motivates to ensure the portfolio is covered the closer we get to the final timestep.
    # prediction.
    latest_news = env.predictions[-1] if env.predictions else 0
    # sum purchased
    purchased_already = np.array([p.mWh for p in env.purchases]).sum()
    needed = latest_news + purchased_already
    action = env.actions[-1] if env.actions else [0,0]
    if needed == 0:
        #absolute punishmeht per action if prediction is 0
        return -abs((action[0] - needed)) * (env._step/ cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL)
    else:
        #normalized by missing to avoid large negative rewards when lots of customers present
        return -abs((action[0] - needed) / needed) * (env._step/ cfg.WHOLESALE_OPEN_FOR_TRADING_PARALLEL)
    # actual function
    #     neg diff between bought and needed, normalized by needed, weighted by "urgency"
    #     --> large diff --> more negative reward



#def shifting_balancing_price(env:PowerTacEnv):
#    """Gives back a relation between the average market price for the target timeslot and the average price the broker achieved"""
#    market_trades = [[tr.executionMWh, tr.executionPrice] for tr in env.cleared_trades]
#    purchases = [[p.mWh, p.price] for p in env.purchases]
#    realized_usage = env.realized_usage
#
#
#    average_market = calculate_running_averages(np.array([market_trades]))[0]
#
#    balancing_needed = calculate_balancing_needed(purchases, realized_usage)
#
#    # TODO for now just a fixed punishment for every balanced mWh. Later maybe based on balancing stats data
#    du_trans = calculate_du_fee(average_market, balancing_needed)
#    if du_trans:
#        purchases.append(du_trans)
#
#    type_, average_agent = average_price_for_power_paid(purchases)
#    market_relative_prices = 0
#    if type_ == 'ask':
#        # broker is overall selling --> higher is better
#        market_relative_prices = average_agent / average_market
#    if type_ == 'bid':
#        #broker is overall buyer --> lower is better
#        market_relative_prices = average_market / average_agent
#
#    #reward is made up of several components
#    # priority one: getting the mWh that are predicted
#    # priority number two: getting those cheaply
#
#    #if balancing high --> ratio close to 1
#    bn_abs = abs(balancing_needed)
#    weight_balancing = bn_abs / (abs(realized_usage) + bn_abs)
#    weight_price = 1 - weight_balancing
#    # the two components of the reward function
#    balancing_reward_component = weight_balancing * (1 / balancing_needed ** 2)
#    price_reward_component = weight_price * market_relative_prices
#    # making sure we don't have nan's or inf's
#    balancing_reward_component = balancing_reward_component if not math.isnan(balancing_reward_component) else 0
#    price_reward_component = price_reward_component if not math.isnan(price_reward_component) else 0
#    reward = balancing_reward_component + price_reward_component
#    return reward
#    #rdreturn market_relative_prices - (balancing_needed/ (balancing_needed + energy_sum))
