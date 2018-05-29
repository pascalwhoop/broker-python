import unittest
import numpy as np
from agent_components.wholesale.mdp import parse_wholesale_file
from agent_components.wholesale.util import calculate_running_averages
from tests.agent_components.wholesale.test_environment import make_mock_wholesale_data


class TestMdp(unittest.TestCase):
    def test_parse_wholesale_file(self):
        test_file_path = "tests/agent_components/wholesale/test_marketprices.csv"
        with open(test_file_path) as f:
            data = parse_wholesale_file(f)
        self.assertEqual(len(data), 400)
        self.assertEqual(len(data[0]), 27)
        without_intro = [row[3:] for row in data]
        self.assertEqual(np.array(without_intro).shape, (400, 24, 2))

    def test_calculate_running_average(self):
        # test the calculation of the historical running average prices per kWh for target timeslot
        # loading wholesale data into entity
        data_ = [row[3:] for row in make_mock_wholesale_data()]
        averages = calculate_running_averages(np.array(data_))
        # print([row[3:] for row in data])
        assert np.isclose(averages[5], 0.6)
        # self.log_env.calculate_running_average(target_timeslot)
