import unittest
from unittest.mock import patch

import numpy as np

from agent_components import demand
from agent_components.demand import data
from agent_components.wholesale.mdp import PowerTacMDPEnvironment, PowerTacLogsMDPEnvironment


class MagickMock(object):
    pass


class TestPowerTacMDPEnvironment(unittest.TestCase):

    """Testing the powertac MDP adapter for the openAI Gym"""

    def setUp(self):
        self.env = PowerTacMDPEnvironment(360)
        self.log_env = PowerTacLogsMDPEnvironment()
        pass

    def tearDown(self):
        demand.data.clear()
        pass

    def test_init(self):
        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.target_timestep, 360)


    def step_block_until(self):
        # when the environment is stepped, it needs to block until all events have arrived.
        #self.env.step()
        pass


    def test__step_timeslot(self):
        pass
        #self.log_env._step_timeslot()
        #self.assertEqual(self.log_env.active_timeslots[0], 361)
        #self.assertEqual(self.log_env.active_timeslots[-1], 384)

    def test_make_random_game_order(self):
        games = self.log_env._make_random_game_order()
        games = np.array(games)
        self.assertEqual(1, games.min())

    def test_parse_wholesale_file(self):
        test_file_path = "tests/agent_components/wholesale/test_marketprices.csv"
        with open(test_file_path) as f:
            data = self.log_env.parse_wholesale_file(f)
        self.assertEqual(len(data), 400)
        self.assertEqual(len(data[0]), 27)
        without_intro = [row[3:] for row in data]
        self.assertEqual(np.array(without_intro).shape, (400,24,2))

    @patch('agent_components.wholesale.mdp.get_demand_data_values')
    @patch('agent_components.wholesale.mdp.parse_usage_game_log')
    def test_make_data_for_game(self, parse_usage, get_demand):
        get_demand.return_value = np.zeros((31,2))
        # mocking another function. setUp creates new object every time so its' ok
        with patch.object(self.log_env, 'parse_wholesale_file') as mock_parse:
            mock_parse.return_value = [3,4]
            data = self.log_env.make_data_for_game(1)
        #assert that the returned data is equal
        self.assertEqual(data[0], [3,4])
        self.assertEqual(data[1].all(), np.zeros((30,2)).all())


