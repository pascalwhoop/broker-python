"""
singleton data storage that holds the customers demand data in a form that the learner can work with
"""
import csv
import logging
from typing import List, Tuple, Dict

import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import Sequence
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

import util.config as cfg
from communication.grpc_messages_pb2 import PBCustomerBootstrapData, PBTariffTransaction, PBTimeslotComplete
from util.learning_utils import NoneScaler

log = logging.getLogger(__name__)

# Map of customerId -> array of transactions
# this serves as a cache so that after each time step, the total per timestep can
# be calculated (customers might use and produce energy at the same time through different contracts)
tariff_transactions = {}

demand_data = {}
scalers = {} #stores the scalers for each customer by name. Assumes that customers don't change their scale by much across games.




def append_usage(name, sum):
    if name not in demand_data:
        demand_data[name] = []
    demand_data[name].append(sum)


def clear():
    """clears the data after a game"""
    global demand_data, tariff_transactions
    demand_data = {}
    tariff_transactions = {}


def sequence_for_usages(usages: np.array, is_flat, scaler=None) -> Sequence:
    """
    Generates a Sequence for a usages array of a customer
    :param usages:
    :return:
    """
   # let's create a targets array by shifting the original by one
    ys = np.zeros((len(usages), 24))
    for i in range(len(usages)):
        twen4h = usages[i + 1:i + 25]
        #if scaler is not None and len(twen4h) > 0:
        #    twen4h = scaler.inverse_transform(twen4h.reshape(-1,1)).flatten()
        ys[i][0:len(twen4h)] = twen4h

    if is_flat is False:
        usages = usages.reshape(-1, 1)

    return TimeseriesGenerator(usages, ys, length=168, batch_size=32)


def get_demand_data_values():
    return np.array(list(demand_data.values()))


def make_sequences_from_historical(is_flat=True, scaler=None) -> List[Sequence]:
    """
    Generates sequences from historical data
    :param is_flat: whether or not the sequence is actually flat (for dense/logres etc) or (168,1) style shape for LSTM
    :return:
    """
    customer_records = list(demand_data.values())
    sequences = [sequence_for_usages(np.array(usages),is_flat, scaler) for usages in customer_records]
    #customer_records = np.array(customer_records).sum(axis=0)
    #sequences = [sequence_for_usages(np.array(usages),is_flat) for usages in [customer_records]]
    return sequences


def preprocess_data(scaler_type=None) -> Dict[str, object]:
    """processes each customer individually and returns the scalers"""

    for cu in demand_data:
        scaler = get_fresh_scaler(scaler_type)
        data_scaled = scaler.fit_transform(np.array(demand_data[cu]).reshape(-1,1)).flatten()
        demand_data[cu] = data_scaled
        scalers[cu] = scaler
    return scalers

def get_fresh_scaler(scaler_type=None):
    if scaler_type is None:
        scaler_type = cfg.DEMAND_DATA_PREPROCESSING_TYPE
    if scaler_type == 'none':
        return NoneScaler()
    if scaler_type == 'minmax':
        return preprocessing.MinMaxScaler()
    if scaler_type == 'standard':
        return preprocessing.StandardScaler()
    if scaler_type == 'robust':
        return preprocessing.RobustScaler()


def parse_usage_game_log(file_path, pp_type=None):
    clear()
    with open(file_path, 'r') as csvfile:
        for row in csv.DictReader(csvfile, delimiter=','):
            name = row['cust'].strip()
            usage = float(row[' production']) + float(row[' consumption'])
            append_usage(name, usage)
    #clearing users that never use any energy
    for i in list(demand_data.items()):
        if (0 == np.array(i[1])).all():
            demand_data.pop(i[0])

    preprocess_data(pp_type)
    return demand_data

def get_first_timestep_for_file(file_path):
    with open(file_path, 'r') as csvfile:
        for row in csv.DictReader(csvfile, delimiter=','):
            ts = int(row['slot'])
            return ts


class DemandForecasts(object):
    """A collection of demand forecasts for all customers of the broker"""
    def __init__(self, target_timestep, forecasts):
        super(DemandForecasts, self).__init__()
        self.target_timestep = target_timestep
        self.forecasts = forecasts

    def total(self):
        return np.sum(self.forecasts)
