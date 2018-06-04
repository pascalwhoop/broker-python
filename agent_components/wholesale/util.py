import numpy as np

from agent_components.wholesale.mdp import np_high
from util.learning_utils import TbWriterHelper

"""Utility functions for the wholesale trading component"""

tb_writer_helper = TbWriterHelper('mdp_agent')

def calculate_running_averages( known_results: np.array):
    """
    Calculates the running averages of all timeslots.
    """
    averages = []

    for result in known_results:
        avg = calculate_running_average(result)
        if avg is 0 and averages:
            avg = averages[-1]
        averages.append(avg)

    return averages


def calculate_running_average(timeslot_trading_data):
    # data is a 24 item long array of 2 vals each
    # average is sum(price_i * kwh_i) / kwh_total
    sum_total = timeslot_trading_data[:, 0].sum()
    avg = 0
    if sum_total != 0.0:
        avg = (timeslot_trading_data[:, 0] * timeslot_trading_data[:, 1]).sum() / sum_total
    # if avg was not set or was set but to 0 use last average for this timestep
    return avg


def get_sum_purchased_for_ts(purchases) -> float:
    if not purchases:
        return 0
    return np.array(purchases)[:, 0].sum()


def calculate_missing_energy(purchases: float, demand: float):
    """
    Determines the amount of balancing that is needed for the agents actions after it purchased for 24 times.

    :param purchases: array of purchases. Positive means purchases, negative means sales of mWh
    :param demand: demand for that timeslot. Negative means agent needs to buy
    :return: amount of additional mWh the agent requires for its portfolio.
    """
    # if demand > purchases --> positive, otherwise negative
    # *(-1) because balancing needed is supposed to be from perspective of "missing transaction" for broker
    return (-1) * (demand + purchases)


def average_price_for_power_paid(bought):
    """
    calculates the average price per mWh the agent paid in the market. If the agent actually sold more than
    bought, the value becomes negative
    :param bought:
    :return: average price, stupidBoolean --> stupid if it paid money to sell energy on average
    """
    # [:,0] is the mWh purchased and [:,1] is the average price for that amount
    total_energy = 0
    total_money = 0
    for t in bought:
        total_energy += t[0]
        total_money += abs(t[0]) * t[1]

    # if both positive --> broker got energy for free --> 0
    # if both negative --> broker paid and lost energy --> infinite
    if total_energy < 0 and total_money < 0:
        return 'bid',  np_high #max number possible. simulates infinite
    # broker lost energy --> sold it
    if total_energy < 0:
        return 'ask',  abs(total_money / total_energy)
    #broker purchased energy and still made profit
    if total_money > 0 and total_energy > 0:
        return 'bid', 0
    #broker purchased energy
    if total_energy > 0:
        return 'bid', abs(total_money / total_energy)


def calculate_du_fee(average_market, balancing_needed):
    du_trans = []
    if balancing_needed > 0:
        # being forced to buy for 5x the market price! try and get your kWh in ahead of time is what it learns
        du_trans = [balancing_needed, -1 * average_market * 5]
    if balancing_needed < 0:
        # getting only a 0.5 of what the normal market price was
        du_trans = [balancing_needed, 0.5 * average_market]  # TODO to config
    if balancing_needed == 0:
        du_trans = [0, 0]
    return du_trans


def trim_data(demand_data: np.array, wholesale_data: np.array, first_timestep_demand):
    min_dd = first_timestep_demand
    # the wholesale data holds 3 columns worth of metadata (slot, dayofweek,hourofday)
    ws_header = np.array([row[0:3] for row in wholesale_data])
    min_ws = int(ws_header[:, 0].min())
    # getting the first common ts
    starting_timeslot = min_dd if min_dd > min_ws else min_ws

    # trim both at beginning to ensure common starting TS
    if min_dd > min_ws:
        # trim ws
        wholesale_data = wholesale_data[min_dd - min_ws:]
    else:
        # trim other
        demand_data = demand_data[min_ws - min_ws:]

    # now trim both at end to ensure same length
    max_len = len(demand_data) if len(demand_data) < len(wholesale_data) else len(wholesale_data)
    demand_data = demand_data[:max_len - 1]
    wholesale_data = wholesale_data[:max_len - 1]
    return demand_data, wholesale_data


def is_cleared(action, market_data) -> bool:
    # if both positive, the agent it trying to pull a fast one. Nope
    if action[0] > 0 and action[1] > 0:
        return False
    # ignoring amounts in offline learning files, just checking prices
    a = action[1]
    m = market_data[1]
    z = np.zeros(a.shape)

    # market not active, nothing sold/bought
    if market_data[0] == 0:
        return False

    if a > 0:
        # selling for less -> cleared
        return a < abs(m)
    if a < 0:
        # buying for more -> cleared
        return abs(a) > m
    # default, didn't buy anything or no price
    return False


def calculate_balancing_needed(purchases, realized_usage):
    # appending final balancing costs for broker for any missing energy
    if len(purchases) == 0:
        balancing_needed = -1 * realized_usage
    else:
        energy_sum = get_sum_purchased_for_ts(purchases)
        balancing_needed = calculate_missing_energy(energy_sum, realized_usage)
    return balancing_needed


