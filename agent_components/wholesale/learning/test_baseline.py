import numpy as np
import unittest
import util.config as cfg

from agent_components.wholesale.environments.PowerTacWholesaleObservation import PowerTacWholesaleObservation
from agent_components.wholesale.learning.baseline import BaselineTrader
from communication.grpc_messages_pb2 import PBOrderbook, PBMarketTransaction


class TestBaseline(unittest.TestCase):

    def setUp(self):
        self.testable = BaselineTrader()
        pass

    def tearDown(self):
        pass

    def test_forward(self):
        obs = self._make_sample_observation()
        action = self.testable.forward(obs)[0]
        assert action[0] == 30
        #pred now -90
        obs.predictions.append(-90)
        action = self.testable.forward(obs)[0]
        assert action[0] == 70
        #pred now +90
        obs.predictions.append(90)
        action = self.testable.forward(obs)[0]
        assert action[0] == -110
        #purchase something more
        obs.purchases.append(PBMarketTransaction(mWh=-110))
        action = self.testable.forward(obs)[0]
        assert action[0] == 0

    def _make_sample_observation(self):
        return PowerTacWholesaleObservation(
            hist_avg_prices=np.arange(cfg.WHOLESALE_HISTORICAL_DATA_LENGTH),
            step=1,
            orderbooks=[PBOrderbook(clearingPrice=10)],
            purchases=[PBMarketTransaction(price=-10, mWh=20)],
            predictions=[-30, -50],
            cleared_trades=[],
            actions=[]
        )
