import unittest

import pytest
from pydispatch import dispatcher
from unittest.mock import Mock, patch, MagicMock

import numpy as np

from agent_components.demand.learning import data
from agent_components.demand.estimator import Estimator
from communication.grpc_messages_pb2 import *
from communication.pubsub import signals
from util.learning_utils import NoneScaler


class TestDemandLearning(unittest.TestCase):

    def setUp(self):
        data.clear()



    def test_data_make_sequences_from_historical(self):
        data.demand_data[0] = np.arange(0, 1000)
        sequences = data.make_sequences_from_historical(False)
        seq = sequences[0]
        x,y = seq[0]
        assert len(x) == 32
        assert len(y) == 32
        assert len(x[0]) == 168
        assert len(y[0]) == 24
        assert (np.arange(0,168).reshape(-1,1) == x[0]).all()
        assert (np.arange(169,169+24) == y[0]).all()
        assert (np.arange(1,169).reshape(-1,1) == x[1]).all()
        assert (np.arange(170,170+24) == y[1]).all()


class TestEstimator(unittest.TestCase):
    def setUp(self):
        model_mock = Mock()
        model_mock.predict.return_value = np.arange(48).reshape((2,24))
        self.e = Estimator(model_mock)

        test_usages = {}
        self.e.usages['Jim'] = test_usages
        self.e.usages['Pop'] = test_usages
        self.e.customer_counts["Jim"] = 1
        self.e.customer_counts["Pop"] = 1
        self.e.customer_populations["Jim"] = 1
        self.e.customer_populations["Pop"] = 1
        self.e.scalers['Jim'] = NoneScaler()
        self.e.scalers['Pop'] = NoneScaler()
        for i in range(300):
            test_usages[i] = i

    def tearDown(self):
        try:
            self.e.unsubscribe()
        except:
            pass

    def test_add_transaction(self):
        self.e.customer_populations['Tim'] = 1
        self.e.customer_counts['Tim'] = 1
        tx = PBTariffTransaction(txType=CONSUME, kWh=5, customerInfo=PBCustomerInfo(name='Tim'), postedTimeslot=4)
        self.e.handle_usage(tx)
        assert self.e.usages['Tim'][4] == 5
        self.e.handle_usage(tx)
        assert self.e.usages['Tim'][4] == 10
        self.e.handle_tariff_transaction_event(None, None, tx)
        assert self.e.usages['Tim'][4] == 15

    @pytest.mark.skip
    def test_handle_customer_bootstrap_data_event(self):
        bs = PBCustomerBootstrapData(customerName="Jim", netUsage=[1,2,3,4,5,6,7,8,9,10])
        self.e.handle_customer_bootstrap_data_event(None, None, bs)
        assert len(self.e.usages['Jim'].values()) == 10

        model_mock = self.e.model
        model_mock.fit_generator.assert_called()

    @patch('agent_components.demand.estimator.dispatcher')
    def test_process_customer_new_data(self,dispatcher_mock:MagicMock()):

        #set the current TS to some value below the number of usages recorded
        self.e.current_timeslot = 200
        model_mock = self.e.model
        #listen to the prediction events
        self.e.process_customer_new_data()

        #length fits
        assert model_mock.fit.call_args[0][0].shape == (2,168)
        assert model_mock.fit.call_args[0][1].shape == (2,24)

        assert (model_mock.fit.call_args[0][0] == np.arange(200-168-24,200-24)).all()
        assert (model_mock.fit.call_args[0][1] == np.arange(200-24,200)).all()
        #TODO opt > add more precise assert
        dispatcher_mock.send.assert_called()
        preds = dispatcher_mock.send.call_args[1]['msg'][0].predictions
        assert preds[210] == 0.01

    def test_handle_sim_end(self):
        self.e.current_timeslot = 1
        self.e.handle_sim_end(None, None, None)
        assert self.e.current_timeslot == 0

    def test_missing_demand_in_estimator(self):
        #let's assume there's no record of the customer for a given TS. That's not a reason to crash the entire pipeline!
        del self.e.usages['Jim'][140]

        #set the current TS to some value below the number of usages recorded
        self.e.current_timeslot = 200
        #shouldn't crash
        self.e.process_customer_new_data()

    def test_handle_usage(self):
        #overall pop is 10
        self.e.customer_populations["A"] = 10
        #actual customers are 2
        self.e.customer_counts["A"] = 2
        tx = PBTariffTransaction(customerInfo=PBCustomerInfo(name="A"), kWh=10, postedTimeslot=44)
        self.e.handle_usage(tx)
        assert self.e.usages["A"][44] == 50

    def test_handle_customer_change(self):
        msg = PBTariffTransaction(customerCount=10, txType=SIGNUP, customerInfo=PBCustomerInfo(population=100, name="A"))
        self.e.handle_customer_change(msg)
        assert self.e.customer_counts["A"] == 10
        msg = PBTariffTransaction(customerCount=5, txType=WITHDRAW, customerInfo=PBCustomerInfo(population=100, name="A"))
        self.e.handle_customer_change(msg)
        assert self.e.customer_counts["A"] == 5
        msg = PBTariffTransaction(customerCount=5, txType=WITHDRAW, customerInfo=PBCustomerInfo(population=100, name="A"))
        self.e.handle_customer_change(msg)
        assert "A" not in self.e.customer_counts


    def test_store_predictions(self):
        predictions = np.arange(24)
        self.e.current_timeslot = 10
        self.e.store_predictions('Jim', predictions)
        assert self.e.predictions['Jim'][11] == [0]
        assert self.e.predictions['Jim'][11+23] == [23]






