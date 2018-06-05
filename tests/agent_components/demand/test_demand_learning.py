import unittest
from pydispatch import dispatcher
from unittest.mock import Mock, patch

import numpy as np

import util.config as cfg

from agent_components.demand import data
from agent_components.demand.estimator import Estimator
from communication.grpc_messages_pb2 import *
from communication.pubsub import signals
from environment.messages_cache import PBTariffTransactionCache
from util.learning_utils import NoneScaler


class TestDemandLearning(unittest.TestCase):

    def setUp(self):
        data.clear()



    def test_data_make_sequences_from_historical(self):
        data.demand_data[0] = np.arange(0,1000)
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
        self.e = Estimator()
        get_model_mock = Mock()
        model_mock = Mock()
        model_mock.predict.return_value = np.arange(24)
        get_model_mock.return_value = model_mock
        self.e.get_model = get_model_mock

    def tearDown(self):
        self.e.unsubscribe()

    def test_add_transaction(self):
        tx = PBTariffTransaction(txType=CONSUME, kWh=5, customerInfo=PBCustomerInfo(name='Jim'), postedTimeslot=4)
        self.e.add_transaction(tx)
        assert self.e.usages['Jim'][4] == 5
        self.e.add_transaction(tx)
        assert self.e.usages['Jim'][4] == 10
        self.e.handle_tariff_transaction_event(None, None, tx)
        assert self.e.usages['Jim'][4] == 15

    def test_handle_customer_bootstrap_data_event(self):
        bs = PBCustomerBootstrapData(customerName="Jim", netUsage=[1,2,3,4,5,6,7,8,9,10])
        self.e.handle_customer_bootstrap_data_event(None, None, bs)
        assert len(self.e.usages['Jim'].values()) == 10

        self.e.get_model.assert_called()
        model_mock = self.e.get_model('Jim')
        model_mock.fit_generator.assert_called()

    @patch('agent_components.demand.estimator.sequence_for_usages')
    def test_process_customer_new_data(self,sequence_mock):
        sequence_mock.return_value = "seq"
        test_usages = {}
        self.e.usages['Jim'] = test_usages
        self.e.scalers['Jim'] = NoneScaler()
        for i in range(300):
            test_usages[i] = i
        model_mock = self.e.get_model()
        model_mock.predict.return_value = np.arange(24)
        #listen to the prediction events
        listen_mock = Mock()
        dispatcher.connect(listen_mock, signal=signals.COMP_USAGE_EST)
        self.e.process_customer_new_data()
        model_mock.fit_generator.assert_called_with('seq')
        listen_mock.assert_called()

    @patch('agent_components.demand.estimator.store_model_customer_nn')
    def test_handle_sim_end(self, store_mock: Mock):
        self.e.models['Jim'] = 'model'
        self.e.handle_sim_end(None, None, None)
        store_mock.assert_called_once_with('model', 'Jim', 'dense_v2')

    def test_store_predictions(self):
        predictions = np.arange(24)
        self.e.current_timeslot = 10
        self.e.store_predictions('Jim', predictions)
        assert self.e.predictions['Jim'][11] == [0]
        assert self.e.predictions['Jim'][11+23] == [23]






