import unittest

import numpy as np

import util.config as cfg

from agent_components.demand import data
from communication.grpc_messages_pb2 import *


class TestDemandLearning(unittest.TestCase):

    def setUp(self):
        data._clear()

    def test_data_timeslot_complete(self):
        one = PBTariffTransaction(kWh=5, customerInfo=PBCustomerInfo(id=1, name="willie"))
        two = PBTariffTransaction(kWh=10, customerInfo=PBCustomerInfo(id=1, name="willie"))
        three = PBTariffTransaction(kWh=-3, customerInfo=PBCustomerInfo(id=1, name="willie"))
        data.update(one)
        data.update(two)
        data.update(three)

        data.calculate_current_timestep(PBTimeslotComplete(timeslotIndex=1))

        # assert summation is correct
        self.assertTrue(data.demand_data["willie"][-1], 12)
        # assert cache is cleared
        self.assertEqual(len(data.tariff_transactions.values()), 0)

    def test_data_handle_bootstrap(self):
        bootstrap_data = PBCustomerBootstrapData(customerName="willie", netUsage=[1,2,3,4,5,6,7,8,9])
        data.update_with_bootstrap(bootstrap_data)
        # assert that the usage history is equal to the bootstrap data
        self.assertEqual(len(data.demand_data["willie"]), 9)

    def test_data_calculate_training_data(self):
        #fill customer data
        data.update_with_bootstrap(PBCustomerBootstrapData(customerName="willie", netUsage=range(100))) #0-99
        #adding one more
        data.update(PBTariffTransaction(customerInfo=PBCustomerInfo(name="willie"), kWh=101))
        data.calculate_current_timestep(PBTimeslotComplete(timeslotIndex=101))
        training_data = data.calculate_training_data(PBTimeslotComplete(timeslotIndex=101))
        #only one customer
        self.assertEqual(len(training_data.x), 1)
        #length of generated x is one week worth
        self.assertEqual(len(training_data.x[0]), cfg.DEMAND_ONE_WEEK)
        #the first part is empty because we supplied too little
        self.assertEqual(0, np.sum(training_data.x[0][0:67]))
        #the last part is all the previously added bootstrap + the extra being the "final" --> y
        self.assertEqual(np.sum(range(100)), np.sum(training_data.x[0][67:]))



