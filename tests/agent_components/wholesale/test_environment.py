import asyncio
import unittest
from collections import Coroutine, Generator
from unittest.mock import Mock, patch

from agent_components.wholesale.environments.PowerTacEnv import WholesaleEnvironmentManager
from agent_components.wholesale.environments.PowerTacMDPEnvironment import PowerTacMDPEnvironment
from communication.grpc_messages_pb2 import PBMarketTransaction, PBTimeslotUpdate


class TestPowerTacMDPEnvironment(unittest.TestCase):
    def setUp(self):
        self.e = PowerTacMDPEnvironment(360)

    def test_subscribe(self):
        self.e.subscribe()

class TestWholesaleEnvironmentManager(unittest.TestCase):
    def setUp(self):
        self.e = WholesaleEnvironmentManager()

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


