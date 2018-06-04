"""
singleton data storage that holds the customers demand data in a form that the learner can work with
"""
import csv
import logging
from typing import List, Tuple

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


def update(tt: PBTariffTransaction) -> None:
    """
    adding transactions to the data sets. This happens until a timeslot is complete
    at which point the transactions are reduced to a single usage value for that timeslot
    :param tt:
    """
    name = tt.customerInfo.name

    if name not in tariff_transactions:
        tariff_transactions[name] = []

    customer_data = tariff_transactions[name]
    customer_data.append(tt)


def make_usages_for_timestep(tc: PBTimeslotComplete) -> None:
    """
    calculates the sum of usages per timeslot per customer
    :param tc:
    :return:
    """
    global tariff_transactions
    for name in tariff_transactions:
        customer_transactions = tariff_transactions[name]
        usages = map(lambda tt: tt.kWh, customer_transactions)
        sum = 0
        for u in usages:
            sum += u

        append_usage(name, sum)

    # reset the cache
    tariff_transactions = {}
    log.debug("Calculated usage for customers after completed timeslot for {} customers".format(len(demand_data)))


def append_usage(name, sum):
    if name not in demand_data:
        demand_data[name] = []
    demand_data[name].append(sum)


def update_with_bootstrap(msg: PBCustomerBootstrapData):
    if msg.customerName not in demand_data:
        data = []
        data.extend(msg.netUsage)
        demand_data[msg.customerName] = data


def clear():
    """clears the data after a game"""
    global demand_data, tariff_transactions
    demand_data = {}
    tariff_transactions = {}


def reverse_scale(twen4h, scaler):

    pass


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


def preprocess_data(scaler=None, type=None):
    keys = list(demand_data.keys())
    data = np.array(list(demand_data.values()))
    assert 2 == len(data.shape)
    data_scaled = None

    shape = data.shape
    data = data.flatten().reshape((-1, 1))
    if type is None:
        type = cfg.DEMAND_DATA_PREPROCESSING_TYPE
    if scaler is None:
        if type == 'none':
            scaler = NoneScaler()
        if type == 'minmax':
            scaler = preprocessing.MinMaxScaler()
            #data_scaled, scaler = scale_minmax(data)
        if type == 'standard':
            scaler = preprocessing.StandardScaler()
        if type == 'robust':
            scaler = preprocessing.RobustScaler()
        scaler.fit(data)

    data_scaled = scaler.transform(data)
    data_scaled = data_scaled.flatten().reshape(shape)

    if data_scaled is not None:
        for i,key in enumerate(keys):
            demand_data[key] = data_scaled[i]
    return scaler


def parse_usage_game_log(file_path, scaler=None, pp_type=None):
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

    scaler = preprocess_data(scaler, pp_type)
    return demand_data, scaler

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
