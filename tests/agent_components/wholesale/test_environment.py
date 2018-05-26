import random
import unittest
from collections import deque
from unittest.mock import patch

import numpy as np
import util.config as cfg

from agent_components import demand
from agent_components.demand import data
from agent_components.wholesale.mdp import PowerTacLogsMDPEnvironment, PowerTacMDPEnvironment, WholesaleActionSpace, \
    WholesaleObservationSpace


class MagickMock(object):
    pass


class TestPowerTacMDPLogEnvironment(unittest.TestCase):
    """Testing the powertac MDP adapter for the openAI Gym"""

    def setUp(self):
        self.env = PowerTacMDPEnvironment(360)
        self.log_env = PowerTacLogsMDPEnvironment()
        self.log_env.wholesale_data = self.make_mock_wholesale_data()
        self.log_env.demand_data = self.make_mock_demand_data()
        self.log_env.active_timeslots = self.make_mock_active_timeslots(self.log_env.wholesale_data)

    def tearDown(self):
        demand.data.clear()

    def test_init(self):
        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.target_timestep, 360)

    def test_step_block_until(self):
        # when the environment is stepped, it needs to block until all events have arrived.
        # self.env.step()
        #TODO real game environment
        pass

    def test__step_timeslot(self):
        for i in range(24):
            self.log_env.active_timeslots[i] = 0
        self.log_env._step_timeslot()
        assert np.array_equal(list(self.log_env.active_timeslots)[0:23], np.zeros(23))
        assert self.log_env.active_timeslots[-1] == 363+24

    def test_make_random_game_order(self):
        games = self.log_env._make_random_game_order()
        games = np.array(games)
        self.assertEqual(1, games.min())

    def test_calculate_running_average(self):
        # test the calculation of the historical running average prices per kWh for target timeslot
        # loading wholesale data into entity
        data = [row[3:] for row in self.log_env.wholesale_data]
        averages = self.log_env.calculate_running_averages(np.array(data))
        # print([row[3:] for row in data])
        assert np.isclose(averages[5], 0.6)
        # self.log_env.calculate_running_average(target_timeslot)
        pass

    def test_get_market_data_now(self):
        # get's the current trades for the upcoming 24h timeslots. it's the diagonal from the first 24 timeslots from up right to bottom left
        market_data_now = self.log_env.get_market_data_now()
        assert market_data_now.shape == (24, 2)
        assert np.array_equal(market_data_now[0], np.array([1, 0.1], dtype=np.float32))

    def test_translate_action_to_real_world_vals(self):
        # mocked data looks right?
        data = self.log_env.wholesale_data
        assert data[0][3][0] == np.float32(1.0)
        assert data[0][3][1] == np.float32(0.1)
        assert data[5][3][0] == np.float32(6.0)
        assert data[5][3][1] == np.float32(0.6)

        # mock some demand forecasts
        self.log_env.forecasts = range(1, 25)

        # generate some action (24 trades)
        actions = []
        # first half we bid 1/5th the average price and try to buy 2x the forecast amount
        for i in range(12):
            actions.append([1, -0.1])
        # second half we ask 1/5th the average price (sell) twice the forecast amount
        for i in range(12):
            actions.append([-1, 0.1])

        real_actions = self.log_env.translate_action_to_real_world_vals(np.array(actions))
        # first upcoming timeslot, average amount is 1 mWh and price is 0.1 mWh
        # meaning we buy 2 mWh for -0.02
        # print(real_actions)
        assert real_actions[0][0] == 2
        assert real_actions[0][1] == -0.02
        assert real_actions[5][0] == 12
        assert real_actions[5][1] == -0.12  # -0.2 * 6
        assert real_actions[13][0] == -28
        assert real_actions[13][1] == 0.28

        #try also with zeros
        self.log_env.translate_action_to_real_world_vals(np.zeros((24,2)))

    def test_action_space(self):
        action = WholesaleActionSpace().sample()
        assert action[0][0] is not None
        assert np.array(list(action)).shape == (24, 2)

    def test_parse_wholesale_file(self):
        test_file_path = "tests/agent_components/wholesale/test_marketprices.csv"
        with open(test_file_path) as f:
            data = self.log_env.parse_wholesale_file(f)
        self.assertEqual(len(data), 400)
        self.assertEqual(len(data[0]), 27)
        without_intro = [row[3:] for row in data]
        self.assertEqual(np.array(without_intro).shape, (400, 24, 2))

    @patch('agent_components.wholesale.mdp.demand_data.get_demand_data_values')
    @patch('agent_components.wholesale.mdp.demand_data.parse_usage_game_log')
    def test_make_data_for_game(self, parse_usage, get_demand):
        get_demand.return_value = np.zeros((31, 2))
        # mocking another function. setUp creates new object every time so its' ok
        with patch.object(self.log_env, 'parse_wholesale_file') as mock_parse, patch.object(self.log_env,
                                                                                            'trim_data') as mock_trim:
            mock_parse.return_value = [3, 4]
            # usually returns wholesale data and demand data in trimmed forms
            mock_trim.return_value = np.zeros((31, 2)), [3, 4]
            demand_data, ws_data = self.log_env.make_data_for_game(1)
        # assert that the returned data is equal
        assert np.array_equal(np.array(ws_data), np.array([3, 4]))
        assert np.array_equal(demand_data, np.zeros((31, 2)))

    def test_trim_games(self):
        demand_data = np.zeros((20, 2))
        wholesale_data = self.make_mock_wholesale_data()

        # making first rows be a range but starting in different points and being of different length
        demand_data, wholesale_data = self.log_env.trim_data(demand_data=demand_data, wholesale_data=wholesale_data,
                                                             first_timestep_demand=365)
        # assert same length now
        assert len(demand_data) == len(wholesale_data)
        # and same starting point
        assert 365 == int(wholesale_data[0][0])

    def test_observation_space(self):
        os = WholesaleObservationSpace()
        assert os.spaces['required_energy'].shape == (24,)
        assert os.spaces['current_prices'].shape == (24,24,2)

    def test_apply_clearings_to_purchases(self):
        #making the active timeslots a bit smaller to handle it with mock data
        self.log_env.active_timeslots = deque(range(363,367))
        ts = self.log_env.active_timeslots
        #some mock actions
        actions = np.array([
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
        ])
        #some mock cleared mappings
        cleared1 = [True,True,False,False]
        cleared2 = [True,True,True,True]
        market_data = np.arange(48).reshape(24,2)
        self.log_env.apply_clearings_to_purchases(actions, cleared1, market_data)
        assert np.array_equal(self.log_env.purchases[ts[0]][0],  [actions[0][0],market_data[0][1]])
        assert ts[2] in self.log_env.purchases
        assert len(self.log_env.purchases[ts[2]]) == 0
        self.log_env.apply_clearings_to_purchases(actions, cleared2, market_data)
        assert np.array_equal(self.log_env.purchases[ts[2]][0], [actions[2][0], market_data[2][1]])
        assert len(self.log_env.purchases[ts[0]]) == 2

    def test_get_sum_purchased_for_ts(self):
        self.log_env.purchases[1] = [[i, i/10] for i in range(24)]
        sum_ = self.log_env.get_sum_purchased_for_ts(ts=1)
        assert sum_ == 276
        sum_ = self.log_env.get_sum_purchased_for_ts(ts=2)
        assert sum_ == 0

    def test_get_new_forecasts(self):
        self.log_env.get_new_forecasts()
        assert np.array_equal(np.arange(2,26), np.array(self.log_env.forecasts))

    def test_make_observation(self):
        assert np.array_equal(self.log_env.historical_prices, np.zeros((168)))
        # mock demand data
        # mock historical prices
        self.log_env.historical_prices.extend(list(range(168)))
        # mock forecasts
        self.log_env.forecasts.extend(list(range(24)))
        #mock purchases
        self.log_env.purchases[365] = [[20, 0.1]]
        self.log_env.purchases[370] = [[12,  0.1]]


        #assert that
        obs = self.log_env.make_observation()
        assert obs['required_energy'].shape == (24,)
        assert obs['required_energy'][3] == 3.0
        assert obs['required_energy'][2] == -18
        assert obs['required_energy'][7] == 7-12
        assert obs['historical_prices'].shape == (168,)
        assert obs['current_prices'].shape == (24, 24, 2)
        # latest in the historicals needs to be timeslot NOW, and first needs to be removed
        # from historicals 359-... 167x + 1 of NOW...
        #TODO

    def test_get_current_knowledge_horizon(self):
        horizon = self.log_env.get_current_knowledge_horizon()
        for i in range(24):
            unknown = np.array(horizon[i])[23 - i:]
            zeros = np.zeros((i+1, 2))
            assert  unknown.shape == zeros.shape
            assert np.array_equal(unknown, zeros)

    def test_append_historical_price(self):

        with patch.object(self.log_env, 'new_game') as new_game_mock:
            for i in range(24):
                self.log_env.append_historical_price()
                self.log_env._step_timeslot()
            new_game_mock.assert_not_called()
            # we should have 24 sums of usage in the historicals now.
            historicals_np = np.array(self.log_env.historical_prices)
            assert np.array_equal(historicals_np[0:-24], np.zeros((cfg.WHOLESALE_HISTORICAL_DATA_LENGTH - 24)))
            assert (historicals_np[-24] != 0).any(), "last 24 are supposed to be not 0"




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
        clearings = self.log_env.which_are_cleared(actions, market_closings)
        assert clearings[0]
        assert not clearings[1]
        assert clearings[2]
        assert not clearings[3]

    def test_new_game(self):
        with patch.object(self.log_env, 'make_data_for_game') as mock_make_data, \
                patch.object(self.log_env, 'step') as mock_step ,  \
                patch.object(self.log_env, 'get_new_forecasts') as mock_make_fc:
            mock_make_data.return_value = ['b'], np.array(list(range(360, 360 + 24)) * 2).reshape((24, 2)),
            mock_step.return_value = 'a', 'b', 'c', 'd'
            obs = self.log_env.new_game()
        assert obs == 'a'
        assert len(self.log_env.active_timeslots) == 24
        mock_step.assert_called_once()
        mock_make_fc.assert_called_once()

        # it should reset the active timeslots
        # it should get new data
        # if games are empty recreate a game list and start from beginning (of ranomized list of games)
        # call step once

        pass

    def test_reset_full(self):
        # self.log_env.reset()
        pass

    def test_average_price_for_power_paid(self):
        in_ = np.arange(6).reshape((3,2))
        in_[:,1] = in_[:,1] * -1
        avg, stupid = self.log_env.average_price_for_power_paid(in_)
        np.testing.assert_almost_equal(avg, 4.3333, decimal=4)
        assert stupid == False


        

    def test_calculate_reward(self):
        #market price average is 0.1 per kWh
        mock_purchases = []
        self.log_env.purchases[363] = mock_purchases

        with patch.object(self.log_env, 'calculate_squared_diff') as squared_diff_mock:
            # mocking the squared diff to be always 0
            squared_diff_mock.return_value = 0

            mock_purchases.append([5,-0.1]) # same as market
            self.log_env.demand_data[0] = -5 #making demand equal to purchases --> no DU balancing
            reward = self.log_env.calculate_reward()
            np.testing.assert_almost_equal(reward, 1)

            # let's get some balancing happening
            self.log_env.demand_data[0] = -10 #10 demand, 5 bought, 5 punishment
            reward = self.log_env.calculate_reward()
            np.testing.assert_almost_equal(reward, 0.333, decimal=3)
            #removing the additional DU balancing
            mock_purchases.pop()


            #purchasing something for too high a price
            mock_purchases.append([5,-0.5]) # same as market
            self.log_env.demand_data[0] = -10
            reward = self.log_env.calculate_reward()
            np.testing.assert_almost_equal(reward, 0.333, decimal=3)

            #now selling some energy, should be back to as before
            mock_purchases.append([-5,0.5]) # same as market
            self.log_env.demand_data[0] = -5
            reward = self.log_env.calculate_reward()
            np.testing.assert_almost_equal(reward, 1, decimal=3)

            #now selling even more energy, net average for broker is negative now
            #it bought energy first then sold it for more. That's a good thing to observe and gets rewarded
            #because the average after the whole round is sold 5kWh for 0.45
            mock_purchases.append([-10,0.5]) # same as market
            self.log_env.demand_data[0] = 5
            reward = self.log_env.calculate_reward()
            np.testing.assert_almost_equal(reward, 8.999, decimal=3)



    # ---------------------------------------------------------------------------------------------
    # helpers and generators below

    def make_mock_demand_data(self):
        return list(range(1,50))

    def make_mock_active_timeslots(self, data):
        # mock the active timesteps
        return deque([row[0] for row in data][:24], maxlen=24)

    def make_mock_wholesale_data(self):
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
