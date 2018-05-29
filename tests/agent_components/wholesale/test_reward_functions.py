import unittest

from agent_components.wholesale.environments.PowerTacLogsMDPEnvironment import PowerTacLogsMDPEnvironment
from agent_components.wholesale.learning.reward_functions import direct_cash_reward, step_close_to_prediction_reward


class TestRewardFunctions(unittest.TestCase):


    def test_direct_cash_reward(self):
        #going through the various cases
        #agent buys 5 under market value. good
        reward = direct_cash_reward(None, None, market_trades=[[10,10]], purchases=[[5, -5]], realized_usage=-5)
        assert reward == 25
        #agent buys but still gains cash, crazy good but rare
        reward = direct_cash_reward(None, None, market_trades=[[10,10]], purchases=[[5, 5]], realized_usage=-5)
        assert reward == 75
        #agent looses money on trading, bad
        reward = direct_cash_reward(None, None, market_trades=[[10,10]], purchases=[[-5, -5]], realized_usage=-5)
        # made 25 loss selling energy it needed and needs balancing 10*50 for balancing and 25 loss --> -575
        # made 25 loss --> -25. Would have made -50 loss on avg --> +25, need -500 balancing --> -475
        assert reward == -475
        #agent sells energy but actually shouldn't, bad
        reward = direct_cash_reward(None, None, market_trades=[[10,10]], purchases=[[-5, 5]], realized_usage=-5)
        #selling 5 energy for +25. should have bought for -50 --> 75. Balancing -500 --> -425
        assert reward == -425

        #buying 5 energy to cover the market demand but for 250 instead of 50 --> -200
        reward = direct_cash_reward(None, None, market_trades=[[10,10]], purchases=[[5, -50]], realized_usage=-5)
        assert reward == -200
        # selling 5x the market price --> +250. Now having to buy 10 for 50 -> -250. Should have paid 50 --> -200
        reward = direct_cash_reward(None, None, market_trades=[[10,10]], purchases=[[-5, 50]], realized_usage=-5)
        assert reward == -200


    def test_step_close_to_prediction_reward(self):
        env = PowerTacLogsMDPEnvironment()
        env.latest_observation = [5, 12]
        action = [3, -4]
        reward = step_close_to_prediction_reward(env, action )
        assert reward == 4
        env.latest_observation = [-5, 12]
        action = [3, -4]
        reward = step_close_to_prediction_reward(env, action )
        assert reward == 8**2
