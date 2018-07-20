import asyncio
import unittest
from collections import Coroutine, Generator
from typing import List
from unittest.mock import Mock, patch

from pydispatch import dispatcher

from agent_components.demand.estimator import CustomerPredictions
from agent_components.wholesale.environments.PowerTacEnv import PowerTacEnv
from agent_components.wholesale.environments.PowerTacWholesaleAgent import PowerTacWholesaleAgent
from agent_components.wholesale.environments.PowerTacWholesaleObservation import PowerTacWholesaleObservation
from agent_components.wholesale.environments.WholesaleEnvironmentManager import WholesaleEnvironmentManager
from agent_components.wholesale.environments.PowerTacMDPEnvironment import PowerTacMDPEnvironment
from agent_components.wholesale.learning.baseline import BaselineTrader
from communication.grpc_messages_pb2 import PBMarketTransaction, PBTimeslotUpdate, PBClearedTrade, PBMarketBootstrapData
import numpy as np

from communication.pubsub import signals


class TestPowerTacMDPEnvironment(unittest.TestCase):
    def setUp(self):
        self.e = PowerTacMDPEnvironment(360)

    def test_subscribe(self):
        self.e.subscribe()

class TestWholesaleEnvironmentManager(unittest.TestCase):
    def setUp(self):
        self.agent_mock = Mock()
        self.reward_mock = Mock()
        self.reward_mock.return_value = 0
        self.env_mgr = WholesaleEnvironmentManager(agent=self.agent_mock, reward_function=self.reward_mock)

    def test_handle_market_transaction(self):
        trans = PBMarketTransaction(timeslot=1, mWh=2, price=-12)
        env_mock = Mock()
        self.env_mgr.environments[1] = env_mock
        self.env_mgr.handle_market_transaction(None, None, trans)
        env_mock.handle_market_transaction.assert_called_with(trans)
        #should err if the env hasn't been added yet
        self.assertRaises(Exception, self.env_mgr.handle_market_transaction, [None, None, PBMarketTransaction(timeslot=2)])

    def test_handle_timeslot_update(self):
        update = PBTimeslotUpdate(firstEnabled=1, lastEnabled=24)
        self.env_mgr.environments[0] = Mock()
        self.env_mgr.environments[0].predictions = [1]
        self.env_mgr.environments[0].purchases = [PBMarketTransaction(mWh=1, price=2)]
        self.env_mgr.environments[0].actions = [[1,2]]
        self.env_mgr.agent = "agent"
        with patch.object(self.env_mgr, 'get_historical_prices') as hp_mock:
            hp_mock.return_value = [1,2,3]
            self.env_mgr.handle_timeslot_update(None, None, update)
    #adds the new ones
        assert 1 in self.env_mgr.environments
        assert 24 in self.env_mgr.environments
        #removes the old ones
        assert 0 not in self.env_mgr.environments
        # historicals added to the new ones
        assert list(self.env_mgr.environments[1]._historical_prices) == [1, 2, 3]
        # agent set on the new ones
        assert self.env_mgr.environments[1].agent == "agent"
        assert len(self.env_mgr.environments.keys()) == 24


    def test_historical_prices(self):
        for i in range(200):
            self.env_mgr.append_historical(PBClearedTrade(timeslot=i, executionPrice=i, executionMWh=1))
        #one extra to show the averaging works
        self.env_mgr.append_historical(PBClearedTrade(timeslot=198, executionPrice=1, executionMWh=3))
        #for the timeslot just after all historical pricesj
        res = self.env_mgr.get_historical_prices(200)
        assert len(res) == 168
        assert res[0] == 200-168
        assert res[-1] == 199
        #the extra one that was added above makes it on average 100
        assert res[-2] == 50.25

        #in the bootstrap situation, we have 336 timeslots and for some reason start at 360. Therefore, it's 24h "lost"
        res = self.env_mgr.get_historical_prices(200 + 24)
        assert(len(res)) == 168
        assert res[-1] == res[-24]
        assert res[0] == 224-168


    def test_handle_predictions(self):
        self.agent_mock.forward.return_value = ([0,0], None, None)
        #create some active timeslots --> active environments
        with patch.object(self.env_mgr, 'get_historical_prices') as hp_mock:
            hp_mock.return_value = np.zeros(168)
            self.env_mgr.handle_timeslot_update(None, None, PBTimeslotUpdate(firstEnabled=169, lastEnabled=169 + 24))
        #some mock preds
        preds:List[CustomerPredictions] = []
        for i in range(3):
            cp = CustomerPredictions("jim{}".format(i), np.arange(24), 169)
            preds.append(cp)
        #call
        self.env_mgr.handle_predictions(None, None, preds)
        #assert
        #assert some orders being sent to server via submitservice
        arg = self.agent_mock.forward.call_args
        assert isinstance(arg[0][0], PowerTacEnv)

    def test_handle_cleared_trade(self):
        self.env_mgr.handle_timeslot_update(None, None, PBTimeslotUpdate(firstEnabled=1, lastEnabled=1))
        msg = PBClearedTrade(timeslot=1, executionMWh=2, executionPrice=3)
        self.env_mgr.handle_cleared_trade(None, None, msg)
        self.env_mgr.handle_cleared_trade(None, None, msg)
        self.env_mgr.handle_cleared_trade(None, None, msg)
        assert len(self.env_mgr.historical_average_prices[1]) == 3


    def test_get_sums_from_preds(self):
        preds = []
        for i in range(5):
            vals = np.zeros(24)
            vals.fill(i)
            pred = CustomerPredictions("john", vals, first_ts=1)
            preds.append(pred)
        sums = self.env_mgr.get_sums_from_preds(preds)
        expected = {i: 10 for i in range(1, 25)}
        for i in expected:
            assert expected[i] == sums[i]

    def test_handle_market_bootstrap_data(self):
        mWh = np.arange(360)
        price = np.arange(360) * 10
        mbd = PBMarketBootstrapData(mwh=mWh, marketPrice=price)
        self.env_mgr.handle_market_bootstrap_data(None, None, mbd)
        for i in range(360):
            assert self.env_mgr.historical_average_prices[i][0][0] == i
            assert self.env_mgr.historical_average_prices[i][0][1] == i * 10




    def test_multiple_coroutines(self):
        """A test for myself. Learning how to use coroutines."""
        cr = []
        received = []

        #creating two callables and calling one from other
        def callable2():
            observation = (yield)
            received.append(observation)

        def callable1():
            return callable2()

        #run this a couple times, I wanna see if I can have many generators
        for i in range(5):
            coro = callable1()
            cr.append(coro)

        #assume they are all generators
        for i in range(5):
            assert isinstance(cr[i], Generator)

        #now let's pass them all some observations
        for i in range(5):
            assert len(received) < i+1
            next(cr[i])
            try:
                cr[i].send(i)
            except StopIteration as e:
                #a generator throws a StopIteration when it is completed
                pass

            assert len(received) == i+1

class TestPowerTacEnv(unittest.TestCase):
    def setUp(self):
        self.testable: PowerTacEnv = PowerTacEnv(Mock(), target_ts=360, historical_prices=np.arange(168))



