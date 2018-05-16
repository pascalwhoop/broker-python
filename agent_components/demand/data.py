"""
singleton data storage that holds the customers demand data in a form that the learner can work with
"""
import logging

import numpy as np

import util.config as cfg

from communication.grpc_messages_pb2 import PBTariffTransaction, PBTimeslotComplete, PBCustomerBootstrapData

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


def calculate_current_timestep(tc: PBTimeslotComplete) -> None:
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

        if name not in demand_data:
            demand_data[name] = []

        demand_data[name].append(sum)

    # reset the cache
    tariff_transactions = {}
    log.debug("Calculated usage for customers after completed timeslot for {} customers".format(len(demand_data)))


def update_with_bootstrap(msg: PBCustomerBootstrapData):
    if msg.customerName not in demand_data:
        data = []
        data.extend(msg.netUsage)
        demand_data[msg.customerName] = data

def _clear():
    global demand_data, tariff_transactions
    demand_data = {}
    tariff_transactions = {}

class DemandTrainingData():
    def __init__(self, ts: PBTimeslotComplete):
        self.x = []  # array of customers usage histories
        self.y = []  # array of customers usage for timeslot
        self.ts: PBTimeslotComplete = ts

    def add_customer(self, x, y):
        self.x.append(x)
        self.y.append(y)


def calculate_training_data(ts: PBTimeslotComplete) -> DemandTrainingData:
    training_data = DemandTrainingData(ts)
    for c in demand_data:
        customer_data = demand_data[c]
        historical = customer_data[-cfg.DEMAND_ONE_WEEK + 1:-1]

        # usage historical length is always the same. if we have not enough data, we pad it
        x = np.zeros((cfg.DEMAND_ONE_WEEK))
        offset = cfg.DEMAND_ONE_WEEK - len(historical)
        x[offset:] = historical

        y = customer_data[-1]
        training_data.add_customer(x, y)
    return training_data


