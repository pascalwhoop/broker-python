import unittest

from agent_components.wholesale.learning.reward_functions import direct_cash_reward


class TestRewardFunctions(unittest.TestCase):


    def test_direct_cash_reward(self):
        #going through the various cases
        #agent buys 5 under market value. good
        reward = direct_cash_reward(None, market_trades=[[10,10]], purchases=[[5, -5]], realized_usage=-5)
        assert reward == 25
        #agent buys but still gains cash, crazy good but rare
        reward = direct_cash_reward(None, market_trades=[[10,10]], purchases=[[5, 5]], realized_usage=-5)
        assert reward == 75
        #agent looses money on trading, bad
        reward = direct_cash_reward(None, market_trades=[[10,10]], purchases=[[-5, -5]], realized_usage=-5)
        # made 25 loss selling energy it needed and needs balancing 10*50 for balancing and 25 loss --> -575
        # made 25 loss --> -25. Would have made -50 loss on avg --> +25, need -500 balancing --> -475
        assert reward == -475
        #agent sells energy but actually shouldn't, bad
        reward = direct_cash_reward(None, market_trades=[[10,10]], purchases=[[-5, 5]], realized_usage=-5)
        #selling 5 energy for +25. should have bought for -50 --> 75. Balancing -500 --> -425
        assert reward == -425

