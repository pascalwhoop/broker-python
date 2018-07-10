import unittest
from collections import deque
from unittest.mock import patch

import numpy as np

from agent_components.demand.learning import data
from agent_components.wholesale.environments.PowerTacLogsMDPEnvironment import PowerTacLogsMDPEnvironment
from agent_components.wholesale.environments.PowerTacMDPEnvironment import PowerTacMDPEnvironment
from agent_components.wholesale.environments.PowerTacEnv import WholesaleObservationSpace, WholesaleActionSpace
from agent_components.wholesale.util import average_price_for_power_paid, is_cleared, trim_data, demand_scaler


class MagickMock(object):
    pass


def make_mock_demand_data():
    return list(range(1, 50))


def make_mock_active_timeslots(data):
    # mock the active timesteps
    return deque([row[0] for row in data][:24], maxlen=24)


def make_mock_wholesale_data():
    # creating wholesale style mock data
    wholesale_data_header = np.zeros((50, 3), dtype=np.int32)
    wholesale_data_header[:, 0] = np.arange(363, 363 + 50).transpose()

    data_core = np.zeros((50, 24, 2), dtype=np.float32)
    # iterate over the rows
    for i in range(len(data_core)):
        # and each market clearing for each of the 24 times the ts was traded
        for j in range(len(data_core[i])):
            # mwh to full numbers
            data_core[i][j][0] = i + 1
            # price to 1/10th that
            data_core[i][j][1] = (i + 1) / 10

    wholesale_data = []
    for i in range(50):
        row = []
        row.extend(wholesale_data_header[i])
        row.extend(list(data_core[i]))
        wholesale_data.append(row)
    return wholesale_data


def make_mock_averages():
    # same as wholesale_data, 50 entries with averages being 1/10th the index+1
    return [i / 10 for i in range(1, 51)]

    pass


class TestPowerTacMDPLogEnvironment(unittest.TestCase):
    """Testing the powertac MDP adapter for the openAI Gym"""

    def setUp(self):
        self.env = PowerTacMDPEnvironment(360)
        self.log_env = PowerTacLogsMDPEnvironment()
        self.log_env.wholesale_averages = make_mock_averages()
        self.log_env.wholesale_data = make_mock_wholesale_data()
        self.log_env.demand_data = make_mock_demand_data()
        self.log_env.active_target_timeslot = self.log_env.wholesale_data[0][0]
        import util.config as cfg
        cfg.WHOLESALE_FORECAST_ERROR_PER_TS = 0

    def tearDown(self):
        data.clear()

    def test_init(self):
        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.target_timestep, 360)


    def test_apply_purchase_if_cleared(self):
        #TODO verify the purchases are created correctly.
        pass


    @patch('agent_components.wholesale.environments.PowerTacLogsMDPEnvironment.step_close_to_prediction_reward')
    def test_step_24times(self, reward_mock):
        for i in range(23):
            obs, rew, done, info = self.log_env.step(np.array([0, 0]))
            assert not done
        obs, rew, done, info = self.log_env.step(np.array([0, 0]))
        assert done

    def test_make_random_game_order(self):
        games = self.log_env._make_random_game_order()
        games = np.array(games)
        self.assertEqual(1, games.min())


    def test_get_market_data_now(self):
        # get's the current trades for the upcoming 24h timeslots. it's the diagonal from the first 24 timeslots from up right to bottom left
        market_data_now = self.log_env.get_market_data_now()
        self.log_env.steps = 0
        assert market_data_now.shape == (2,)
        np.testing.assert_almost_equal(market_data_now, np.array([1, 0.1]), decimal=3)

    @patch('agent_components.wholesale.environments.PowerTacLogsMDPEnvironment.price_scaler.inverse_transform')
    def test_translate_action_to_real_world_vals(self, scaler_mock):
        # this makes all results of the inverse_transform be 5. So all [1] actions are multiplied with this
        scaler_mock.return_value = 5
        self.log_env.latest_observeration = [1]
        # mocked data looks right?
        data = self.log_env.wholesale_data
        assert data[0][3][0] == np.float32(1.0)
        assert data[0][3][1] == np.float32(0.1)
        assert data[5][3][0] == np.float32(6.0)
        assert data[5][3][1] == np.float32(0.6)

        # mock some demand forecasts --> demand of 1 in step 1
        self.log_env.forecasts = list(range(1, 25))

        # first upcoming timeslot, latest amount is 1 mWh and price is 0.1 mWh
        # meaning we buy 2 mWh for -0.02
        # print(real_actions)
        self.log_env.steps = 1
        real_action = self.log_env.translate_action_to_real_world_vals(np.array([1, -0.1]))
        np.testing.assert_almost_equal(real_action, [-1, 5 * -0.1])
        real_action = self.log_env.translate_action_to_real_world_vals(np.array([1, -0.5]))
        np.testing.assert_almost_equal(real_action, [-1, -2.5])
        self.log_env.demand_data[0] = 14
        real_action = self.log_env.translate_action_to_real_world_vals(np.array([-1, 1.0]))
        np.testing.assert_almost_equal(real_action, [14, 5 * 1.0])

        # try also with zeros
        self.log_env.translate_action_to_real_world_vals(np.zeros((2)))

    def test_action_space(self):
        action = WholesaleActionSpace().sample()
        assert action.shape == (2,)

    def test_make_observation(self):
        """
        Assuming that this is a flat array of all data that is part of an observation
        :return:
        """
        obs0 = demand_scaler.transform((-1) * self.log_env.demand_data[0]).flatten()[0]
        obs = self.log_env.make_observation()
        assert obs[0] == obs0
        assert len(obs) == 1 + 24

    @patch('agent_components.wholesale.environments.PowerTacLogsMDPEnvironment.trim_data')
    @patch('agent_components.wholesale.environments.PowerTacLogsMDPEnvironment.parse_wholesale_file')
    @patch('agent_components.wholesale.environments.PowerTacLogsMDPEnvironment.demand_data.get_demand_data_values')
    @patch('agent_components.wholesale.environments.PowerTacLogsMDPEnvironment.demand_data.parse_usage_game_log')
    def test_make_data_for_game(self, parse_usage_mock, get_demand_mock, mock_parse_wholesale, mock_trim):
        get_demand_mock.return_value = np.zeros((31, 2))
        # mocking another function. setUp creates new object every time so its' ok
        mock_parse_wholesale.return_value = [3, 4]
        # usually returns wholesale data and demand data in trimmed forms
        mock_trim.return_value = np.zeros((31, 2)), [3, 4]
        demand_data, ws_data = self.log_env.make_data_for_game(1)
        # assert that the returned data is equal
        assert np.array_equal(np.array(ws_data), np.array([3, 4]))
        assert np.array_equal(demand_data, np.zeros((31, 2)))

    def test_trim_games(self):
        demand_data = np.zeros((20, 2))
        wholesale_data = make_mock_wholesale_data()

        # making first rows be a range but starting in different points and being of different length
        demand_data, wholesale_data = trim_data(demand_data=demand_data, wholesale_data=wholesale_data,
                                                             first_timestep_demand=365)
        # assert same length now
        assert len(demand_data) == len(wholesale_data)
        # and same starting point
        assert 365 == int(wholesale_data[0][0])

    def test_observation_space(self):
        os = WholesaleObservationSpace()
        assert os.shape == (25,)

    def test_get_sum_purchased_for_ts(self):
        self.log_env.purchases = [[i, i / 10] for i in range(24)]
        sum_ = self.log_env.get_sum_purchased_for_ts()
        assert sum_ == 276

    def test_get_current_knowledge_horizon(self):
        self.log_env.steps = 5
        horizon = self.log_env.get_current_knowledge_horizon()
        unknown = np.array(horizon[23 - 5:])
        zeros = np.zeros((5 + 1, 2))
        assert unknown.shape == zeros.shape
        assert np.array_equal(unknown, zeros)

    def test_which_are_cleared(self):
        market_closings = np.array([
            [20, 0.1],
            [20, 0.2],
            [20, 0.3],
            [20, 0.4]

        ])
        actions = np.array([
            [-1, 0.09],  # selling for cheap     --> clearing
            [-1, 5.0],  # selling too expensive --> not clearing
            [1, -3.0],  # buying super expensive --> clearing
            [1, -0.3],  # buying but too cheap --> not clearing
        ])
        assert is_cleared(actions[0], market_closings[0])
        assert not is_cleared(actions[1], market_closings[1])
        assert is_cleared(actions[2], market_closings[2])
        assert not is_cleared(actions[3], market_closings[3])

    @patch('agent_components.wholesale.environments.PowerTacLogsMDPEnvironment.cfg')
    def test_get_new_forecast(self, mock_cfg):
        mock_cfg.WHOLESALE_FORECAST_ERROR_PER_TS = 0
        self.log_env.demand_data[0] = 10
        fc = self.log_env.get_forecast_for_active_ts()
        assert fc == 10
        mock_cfg.WHOLESALE_FORECAST_ERROR_PER_TS = 0.01
        fc = self.log_env.get_forecast_for_active_ts()
        assert fc < 12.4 and fc > 7.6 and fc != 10

    def test_apply_wholesale_averages(self):
        wd = [
            [1, 2, 3, [1, 1], [1, 1]],
            [2, 2, 3, [1, 1], [1, 3]]
        ]
        self.log_env.apply_wholesale_averages(wd)
        assert self.log_env.wholesale_averages[1] == 1
        assert self.log_env.wholesale_averages[2] == 2

    def test_new_game(self):
        # TODO
        pass

    def test_reset(self):
        ts = self.log_env.active_target_timeslot
        len_demand = len(self.log_env.demand_data)
        len_ws = len(self.log_env.wholesale_data)
        self.log_env.purchases = [1, 2, 3]

        self.log_env.reset()

        assert self.log_env.steps == 0
        assert ts + 1 == self.log_env.active_target_timeslot
        assert not self.log_env.purchases
        assert len_ws - 1 == len(self.log_env.wholesale_data)
        assert len_demand - 1 == len(self.log_env.demand_data)

        def mock_ng():
            self.log_env.wholesale_data = make_mock_wholesale_data()
            self.log_env.demand_data = make_mock_demand_data()

        with patch.object(self.log_env, 'new_game') as new_game_mock:
            new_game_mock.side_effect = mock_ng

            self.log_env.wholesale_data = []
            self.log_env.demand_data = []
            self.log_env.reset()
            new_game_mock.assert_called_once()

    def test_average_price_for_power_paid(self):
        in_ = np.arange(6).reshape((3, 2))
        in_[:, 1] = in_[:, 1] * -1
        type, avg = average_price_for_power_paid(in_)
        np.testing.assert_almost_equal(avg, 4.3333, decimal=4)
        assert type == 'bid'

        sample_bought = [[-346.49495303, 0.96304058], [81.39146869, 1.38586367],
                         [66.65719033, 8.75573683], [-418.55988655, 8.72186681],
                         [-508.60194753, 12.92631053], [-360.21051942, 16.94025098],
                         [449.23446341, 17.18214492], [599.71790147, 17.65111545],
                         [572.6739119, 16.60269042], [1066.79119908, 15.53783209],
                         [1537.3969027, 13.1445373], [1035.57630062, 16.3898943],
                         [561.39388178, 18.85961063], [829.36020626, 16.32019667],
                         [1435.38659065, 16.86783396], [1774.03794508, -36.41719069],
                         [877.50581612, -38.14916018], [1050.94563103, -31.22096841],
                         [1337.94307139, -27.76994933], [1576.01239402, -27.76967208],
                         [1352.12704343, -26.64545064], [-8762.585611457187, 24.24638286233939]]
        #TODO with sample data

    def test_is_cleared(self):
        #TODO make sure not clearing anything wrong
        pass



