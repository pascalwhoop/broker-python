import unittest
from unittest.mock import Mock, patch, mock_open

import util.config as cfg
import numpy as np

from agent_components.wholesale.environments.LogEnvManagerAdapter import LogEnvManagerAdapter
from agent_components.wholesale.environments.test_log_environment import make_mock_wholesale_data
from agent_components.wholesale.learning.baseline import BaselineTrader
from communication.grpc_messages_pb2 import PBOrder
from communication.pubsub import signals


class TestLogEnvManagerAdapter(unittest.TestCase):

    def setUp(self):
        self.mock_agent = Mock()
        self.testable = LogEnvManagerAdapter(self.mock_agent)



    @patch('agent_components.wholesale.environments.LogEnvManagerAdapter.parse_wholesale_file')
    def test_make_wholesale_data(self,parse_mock):
        parse_mock.return_value = make_mock_wholesale_data()
        with patch("builtins.open", mock_open(read_data="data")) as mock_file:
            self.testable.make_wholesale_data("some")
        for i in range(363, 363+50):
            assert i in self.testable.wholesale_data
            assert (24,2) == self.testable.wholesale_data[i].shape

    @patch('agent_components.wholesale.environments.LogEnvManagerAdapter.demand_data')
    def test_make_demand_data(self,demand_data_mock):
        demand_data_mock.get_demand_data_values.return_value = make_mock_demand_data()
        self.testable.make_demand_data("some")
        assert len(self.testable.demand_data) == 30
        assert len(self.testable.demand_data[0]) == 1200


    @patch('agent_components.wholesale.environments.LogEnvManagerAdapter.dispatcher')
    def test_step_game(self, dispatcher_mock:Mock):
        _fill_with_mock_data(self.testable)
        self.testable.current_timestep = 364
        #mock_wholesale_data gives 50 timesteps --> expect 50 rounds
        self.testable.step_game()
        assert dispatcher_mock.send.call_args_list[0][0][0] == signals.PB_TIMESLOT_COMPLETE
        assert dispatcher_mock.send.call_args_list[1][0][0] == signals.PB_TIMESLOT_UPDATE
        assert dispatcher_mock.send.call_args_list[2][0][0] == signals.COMP_USAGE_EST
        #assert now next  24 messages to be cleared_trades
        assert dispatcher_mock.send.call_args_list[3][0][0] == signals.PB_CLEARED_TRADE
        #and 25th to be a new timeslot_update
        assert dispatcher_mock.send.call_args_list[27][0][0] == signals.PB_TIMESLOT_COMPLETE
        #TODO finish

    def test_fuzz_forecast_for_training(self):
        fc = np.arange(24, dtype=np.float64)
        err = cfg.WHOLESALE_FORECAST_ERROR_PER_TS
        fuzzed = self.testable.fuzz_forecast_for_training(fc)
        #assert that all fuzzed forecasts are within range of acceptable error
        for i in range(24):
            assert (fc[i] - i*err * fc[i]) <= fuzzed[i]
            assert (fc[i] + i*err * fc[i]) >= fuzzed[i]

    @patch('agent_components.wholesale.environments.LogEnvManagerAdapter.dispatcher')
    def test_simulate_predictions(self, dispatcher_mock: Mock):
        _fill_with_mock_data(testable=self.testable)
        self.testable.current_timestep=363
        self.testable.simulate_predictions()
        assert dispatcher_mock.send.call_count == 1
        #assert received a list of 30 CustomerPredictions
        assert len(dispatcher_mock.send.call_args[1]['msg']) == 30


    @patch('agent_components.wholesale.environments.LogEnvManagerAdapter.dispatcher')
    def test_evaluate_orders_received(self, dispatcher_mock):
        _fill_with_mock_data(self.testable)
        self.testable.current_timestep = 363

        self.testable.orders = [PBOrder(timeslot=365, mWh=1, limitPrice=3)]
        self.testable.evaluate_orders_received()
        assert dispatcher_mock.send.call_count == 0

        #really cheap one
        self.testable.orders = [PBOrder(timeslot=365, mWh=-0.001, limitPrice=0.00001)]
        self.testable.evaluate_orders_received()
        assert dispatcher_mock.send.call_count == 1

        #finally, asserting that orders are cleared after handling
        assert not self.testable.orders

    def test_is_cleared_with_volume_probability(self):
        order = PBOrder(mWh=10, limitPrice=-10)
        market = [1000000, 5]
        is_cleared, prob = self.testable.is_cleared_with_volume_probability(order, market)
        assert prob > 0.9 #offering twice the market price and buying very little

        order = PBOrder(mWh=10, limitPrice=-3)
        market = [1000000, 5]
        is_cleared, prob = self.testable.is_cleared_with_volume_probability(order, market)
        assert prob == 0.0 #offering too little

        order = PBOrder(mWh=10, limitPrice=-5.1)
        market = [1000000, 5]
        is_cleared, prob = self.testable.is_cleared_with_volume_probability(order, market)
        assert prob < 0.1 #offering too little



    def test_get_first_timestep(self):
        _fill_with_mock_data(self.testable)
        fts = self.testable.get_first_timestep()
        assert fts == 363





def _fill_with_mock_data(testable:LogEnvManagerAdapter):
    with patch('agent_components.wholesale.environments.LogEnvManagerAdapter.parse_wholesale_file') as parse_mock, \
         patch('agent_components.wholesale.environments.LogEnvManagerAdapter.demand_data') as demand_data_mock, \
         patch("builtins.open", mock_open(read_data="data")) as mock_file:
        parse_mock.return_value = make_mock_wholesale_data()
        demand_data_mock.get_demand_data_values.return_value = make_mock_demand_data()
        testable.make_wholesale_data("some")
        testable.make_demand_data("some")


def make_mock_demand_data():
    return np.arange(31*1200).reshape((31,1200))




