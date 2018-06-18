import asyncio
import unittest
from collections import Coroutine, Generator
from typing import List
from unittest.mock import Mock, patch

from pydispatch import dispatcher

from agent_components.demand.estimator import CustomerPredictions
from agent_components.wholesale.environments.PowerTacEnv import WholesaleEnvironmentManager, PowerTacWholesaleAgent, \
    PowerTacWholesaleObservation
from agent_components.wholesale.environments.PowerTacMDPEnvironment import PowerTacMDPEnvironment
from agent_components.wholesale.learning.baseline import BaselineTrader
from communication.grpc_messages_pb2 import PBMarketTransaction, PBTimeslotUpdate, PBClearedTrade
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
        self.e = WholesaleEnvironmentManager(agent=self.agent_mock)

    def test_handle_market_transaction(self):
        trans = PBMarketTransaction(timeslot=1, mWh=2, price=-12)
        env_mock = Mock()
        self.e.environments[1] = env_mock
        self.e.handle_market_transaction(None, None, trans)
        env_mock.handle_market_transaction.assert_called_with(trans)
        #should err if the env hasn't been added yet
        self.assertRaises(Exception, self.e.handle_market_transaction, [None, None, PBMarketTransaction(timeslot=2)])

    def test_handle_timeslot_update(self):
        update = PBTimeslotUpdate(firstEnabled=1, lastEnabled=24)
        self.e.environments[0] = Mock()
        self.e.agent = "agent"
        with patch.object(self.e, 'get_historical_prices') as hp_mock:
            hp_mock.return_value = [1,2,3]
            self.e.handle_timeslot_update(None, None, update)
        #adds the new ones
        assert 1 in self.e.environments
        assert 24 in self.e.environments
        #removes the old ones
        assert 0 not in self.e.environments
        # historicals added to the new ones
        assert list(self.e.environments[1]._historical_prices) == [1,2,3]
        # agent set on the new ones
        assert self.e.environments[1].agent == "agent"


    def test_historical_prices(self):
        for i in range(200):
            self.e.append_historical(PBClearedTrade(timeslot=i, executionPrice=i, executionMWh=1))
        #one extra to show the averaging works
        self.e.append_historical(PBClearedTrade(timeslot=198, executionPrice=1, executionMWh=3))
        res = self.e.get_historical_prices(200)
        assert len(res) == 168
        assert res[0] == 200-168
        assert res[-1] == 199
        #the extra one that was added above makes it on average 100
        assert res[-2] == 50.25

    def test_handle_predictions(self):
        self.agent_mock.forward.return_value = np.zeros(2)
        #create some active timeslots --> active environments
        with patch.object(self.e, 'get_historical_prices') as hp_mock:
            hp_mock.return_value = np.zeros(168)
            self.e.handle_timeslot_update(None, None, PBTimeslotUpdate(firstEnabled=169, lastEnabled=169+24))
        #some mock preds
        preds:List[CustomerPredictions] = []
        for i in range(3):
            cp = CustomerPredictions("jim{}".format(i), np.arange(24), 169)
            preds.append(cp)
        #call
        self.e.handle_predictions(None, None, preds)
        #assert some orders being sent to server via submitservice
        arg = self.agent_mock.forward.call_args
        assert isinstance(arg[0][0], PowerTacWholesaleObservation)


    def test_get_sums_from_preds(self):
        preds = []
        for i in range(5):
            vals = np.zeros(24)
            vals.fill(i)
            pred = CustomerPredictions("john", vals, first_ts=1)
            preds.append(pred)
        sums = self.e.get_sums_from_preds(preds)
        expected = {i: 10 for i in range(1, 25)}
        for i in expected:
            assert expected[i] == sums[i]




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


