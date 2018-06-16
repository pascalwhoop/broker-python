import unittest
from typing import List

import numpy as np
from unittest.mock import Mock

from pydispatch import dispatcher
from sklearn.preprocessing import MinMaxScaler

from agent_components.demand.estimator import Estimator, CustomerPredictions
from agent_components.wholesale.environments.PowerTacEnv import WholesaleEnvironmentManager
from communication.grpc_messages_pb2 import PBTimeslotComplete, PBTimeslotUpdate
from communication.pubsub import signals

TIMESLOT_NOW = 200

class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.mm = Mock()
        self.mm.predict.return_value = np.arange(0,24, dtype=np.float64).reshape(-1,1)
        self.e = Estimator(self.mm)
        self.e.subscribe()

        #mocking some values in the estimator
        for i in range(TIMESLOT_NOW-1):
            self.e._apply_usage('jim', i,i)
        self.e.scalers['jim'] = MinMaxScaler().fit(np.array([1,1000],dtype=np.float64).reshape(-1,1))

        #create a new env manager
        self.wem = WholesaleEnvironmentManager()
        self.wem.get_historical_prices = Mock()
        self.wem.get_historical_prices.return_value = np.zeros(168)
        self.wem.subscribe()

    def tearDown(self):
        self.e.unsubscribe()
        self.wem.unsubscribe()

    def test_estimator_wholesale_integration(self):
        #listen for predictions calculated
        predictions_received: List[CustomerPredictions] = []
        def listen_pred_ev(signal, sender, msg):
            predictions_received.append(msg)
        dispatcher.connect(listen_pred_ev, signals.COMP_USAGE_EST)
        #listen for wholesale orders
        orders_received = []
        def listen_orders_ev(signal, sender, msg):
            orders_received.append(msg)
        dispatcher.connect(listen_orders_ev, signals.PB_ORDER)

        # 1. send some market messages that get picked up by wholesale
        dispatcher.send(signal=signals.PB_TIMESLOT_UPDATE, msg=PBTimeslotUpdate(firstEnabled=TIMESLOT_NOW, lastEnabled=TIMESLOT_NOW+24))
        # no cleared trades or market transactions necessary for core test
        assert len(orders_received) == 0

        # 2. send pubsub messages that trigger prediction
        dispatcher.send(signal=signals.PB_TIMESLOT_COMPLETE, msg=PBTimeslotComplete(timeslotIndex=TIMESLOT_NOW-1))
        assert len(predictions_received) == 1
        assert len(predictions_received[0].predictions) == 24

        # 3. expect wholesale market to react to prediction
        assert len(orders_received) == 24


        # 4. expect wholesale market to now trigger
        # clean up listeners
        dispatcher.disconnect(listen_pred_ev, signals.COMP_USAGE_EST)
        dispatcher.disconnect(listen_orders_ev, signals.PB_ORDER)



