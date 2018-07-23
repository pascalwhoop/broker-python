import ast
import csv
import logging
import random
from typing import List

import numpy as np
from sklearn import preprocessing

from communication.grpc_messages_pb2 import PBMarketTransaction, PBOrder, PBClearedTrade
from util import config as cfg
from util.config import MIN_PRICE_SCALE, MAX_PRICE_SCALE, MIN_DEMAND, MAX_DEMAND

"""Utility functions for the wholesale trading component"""

log = logging.getLogger(__name__)

def calculate_running_averages(known_results: np.array):
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


def calculate_running_average(timeslot_trading_data: np.array):
    """Assumes an input being an array of [mWH price] items"""
    # data is a 24 item long array of 2 vals each
    # average is sum(price_i * kwh_i) / kwh_total
    s = timeslot_trading_data.shape
    if len(s) == 1 or s[1] != 2:
        return 0
    sum_total = timeslot_trading_data[:, 0].sum()
    avg = 0
    if sum_total != 0.0:
        avg = (timeslot_trading_data[:, 0] * timeslot_trading_data[:, 1]).sum() / sum_total
    return avg


def get_sum_purchased_for_ts(purchases: List[np.array]) -> float:
    """Assumes input as [mWh, price] array"""
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


def average_price_for_power_paid(purchases):
    """
    calculates the average price per mWh the agent paid in the market.
    :param purchases:
    :return: type and average price paid --> stupid if it paid money to sell energy on average
    """
    # [:,0] is the mWh purchased and [:,1] is the average price for that amount
    total_energy = 0
    total_money = 0
    for t in purchases:
        total_energy += t[0]
        total_money += abs(t[0]) * t[1]

    # if both positive --> broker got energy for free --> 0
    # if both negative --> broker paid and lost energy --> infinite
    if total_energy < 0 and total_money < 0:
        return 'bid', cfg.np_high  # max number possible. simulates infinite
    # broker lost energy --> sold it
    if total_energy < 0:
        return 'ask', abs(total_money / total_energy)
    # broker purchased energy and still made profit
    if total_money > 0 and total_energy > 0:
        return 'bid', 0
    # broker purchased energy
    if total_energy > 0:
        return 'bid', abs(total_money / total_energy)
    # if all else fails
    return 'bid', 0


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


#def trim_data(demand_data: np.array, wholesale_data: np.array, first_timestep_demand):
#    :param demand_data:
#    :param wholesale_data:
#    :param first_timestep_demand:
#    :return:
#    """
#    min_dd = first_timestep_demand
#    # the wholesale data holds 3 columns worth of metadata (slot, dayofweek,hourofday)
#    ws_header = np.array([row[0:3] for row in wholesale_data])
#    min_ws = int(ws_header[:, 0].min())
#    # getting the first common ts
#    starting_timeslot = min_dd if min_dd > min_ws else min_ws
#
#    # trim both at beginning to ensure common starting TS
#    if min_dd > min_ws:
#        # trim ws
#        wholesale_data = wholesale_data[min_dd - min_ws:]
#    else:
#        # trim other
#        demand_data = demand_data[min_ws - min_ws:]
#
#    # now trim both at end to ensure same length
#    max_len = len(demand_data) if len(demand_data) < len(wholesale_data) else len(wholesale_data)
#    demand_data = demand_data[:max_len - 1]
#    wholesale_data = wholesale_data[:max_len - 1]
#    return demand_data, wholesale_data


#def is_cleared(action, market_data) -> bool:
#    # if both positive, the agent it trying to pull a fast one. Nope
#    if action[0] > 0 and action[1] > 0:
#        return False
#    # ignoring amounts in offline learning files, just checking prices
#    a = action[1]
#    m = market_data[1]
#    z = np.zeros(a.shape)
#
#    # market not active, nothing sold/bought
#    if market_data[0] == 0:
#        return False
#
#    if a > 0:
#        # selling for less -> cleared
#        return a < abs(m)
#    if a < 0:
#        # buying for more -> cleared
#        return abs(a) > m
#    # default, didn't buy anything or no price
#    return False


def calculate_balancing_needed(purchases, realized_usage):
    # appending final balancing costs for broker for any missing energy
    if len(purchases) == 0:
        balancing_needed = -1 * realized_usage
    else:
        energy_sum = get_sum_purchased_for_ts(purchases)
        balancing_needed = calculate_missing_energy(energy_sum, realized_usage)
    return balancing_needed




def parse_wholesale_file(file):
    out = []
    reader = csv.reader(file)
    for row in reader:
        out.append([ast.literal_eval(str.strip(cell).replace(' ', ',')) for cell in row])
    return out


def _get_wholesale_as_nparr(wholesale_data: List):
    """Assumes it's being passed a list of wholesale data, where the first three columns are metadata and then it's raw stuff"""
    return np.array([row[3:] for row in wholesale_data])


price_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
price_scaler.fit(np.array([MIN_PRICE_SCALE, MAX_PRICE_SCALE]).reshape(-1, 1))
demand_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
demand_scaler.fit(np.array([MIN_DEMAND, MAX_DEMAND]).reshape(-1, 1))


def calculate_energy_needed(latest_news, purchases: List[PBMarketTransaction]):
    purchased_already = np.array([p.mWh for p in purchases]).sum()
    #        what we need *(-1)  + sum purchased (reduces need)
    needed = - latest_news       + purchased_already
    return needed


def is_cleared_with_volume_probability(order:PBOrder, marketClearing: np.array) -> (bool, float):
    """clears the order with a high probability if the offer was very generous and less prob if it is close to the market price"""
    action = np.array([order.mWh, order.limitPrice])
    clearing_probability = 1

    # if both positive, the agent it trying to pull a fast one. Nope
    if action[0] > 0 and action[1] > 0:
        #log.warning("agent is returning irrational bids")
        return False, 0
    if action[0] < 0 and action[1] < 0:
        #log.warning("agent is giving away money")
        return False, 0

    market_volume = abs(marketClearing[0])
    action_volume = abs(action[0])
    if market_volume == 0:
        return False, 0
    if action_volume > market_volume:
        log.warning("agent is asking for large volume {} in market {}".format(action_volume, market_volume))
        return False, 0
    volume_diff = market_volume - action_volume
    #the larger the action_volume, the less likely that the market will clear it.
    clearing_probability *=  (volume_diff / market_volume)


    action_price = action[1]
    market_price = abs(marketClearing[1])

    #selling energy
    if action_price >= 0:
        # selling for less -> cleared
        price_diff = market_price - action_price
    else:
        # buying for more -> cleared
        price_diff = abs(action_price) - market_price

    if price_diff <= 0 or market_price == 0:
        #buying too cheap or selling to expensive
        #or market price is not yet set, because nothing traded
        return False, 0
    price_multiplicator = (price_diff / market_price)
    price_multiplicator = np.array([price_multiplicator, 1]).min()
    clearing_probability *= price_multiplicator

    return random.random() < clearing_probability, clearing_probability


def fuzz_forecast_for_training(customer_data):
    """fuzzes the data a bit according to configured noise factor"""
    for i in range(len(customer_data)):
        err_mult = random.uniform(-cfg.WHOLESALE_FORECAST_ERROR_PER_TS, cfg.WHOLESALE_FORECAST_ERROR_PER_TS)
        # fuzz them increasingly much
        customer_data[i:] = customer_data[i:] + err_mult * customer_data[i:]
    return customer_data