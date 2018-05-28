import math

import numpy as np

from agent_components.wholesale.learning.util import calculate_balancing_needed
from agent_components.wholesale.util import calculate_running_averages, calculate_du_fee, average_price_for_power_paid

# what can rewards be based on?
# - action
# - realized cost
# - divergence from forecast


def simple_truth_ordering(action, market_trades, purchases, realized_usage):
    """
    This helper function trains the agent to initially always order exactly the amount it has forecasted to need and
    offer a price that is always 10% better than the market price --> always clearing
    :param action:
    :return:
    """
    balancing_needed = calculate_balancing_needed(purchases, realized_usage)

    amount = action[0]
    price = action[1]
    mse_amount = (amount-balancing_needed)**2
    reward_price = abs(price - 0.55)
    return 1/(mse_amount + reward_price)


def direct_cash_reward(action, market_trades, purchases, realized_usage):
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



    # if agent paid less than what it would have with average prices --> positive reward



def shifting_balancing_price(action, market_trades, purchases, realized_usage):
    """Gives back a relation between the average market price for the target timeslot and the average price the broker achieved"""
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
