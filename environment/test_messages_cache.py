import pickle

import json
import numpy as np
import os

import communication.pubsub.signals as signals
import unittest
from google.protobuf.json_format import MessageToJson
from unittest.mock import patch, Mock

from agent_components.demand.estimator import CustomerPredictions
from communication.grpc_messages_pb2 import PBTimeslotComplete
from communication.pubsub import signals
import environment.messages_cache as caches

testing_signal = "some_signal"

class TestMessageCache(unittest.TestCase):

    def setUp(self):
        caches._path = "data/testing/"

    def tearDown(self):
        caches._close_all_handlers()
        for f in os.listdir("data/testing/"):
            os.remove(os.path.join("data/testing/", f))

    def test_cache_all_pb_messages(self):
        msg = PBTimeslotComplete(timeslotIndex=2)
        caches.store_message(None, signals.PB_TIMESLOT_COMPLETE, msg)
        self.assertTrue(len(caches.PBTimeslotCompleteCache) > 0)

    def test_log_protobuf_message(self):
        msg = PBTimeslotComplete(timeslotIndex=2)
        MessageToJson(msg)

    @patch('environment.messages_cache.get_file_handler')
    def test_log_protobuf(self,gfhm: Mock):
        file_handler_mock = Mock()
        gfhm.return_value  = file_handler_mock

        msg = PBTimeslotComplete(timeslotIndex=2)
        caches.log_message(signals.PB_TIMESLOT_COMPLETE, msg)
        file_handler_mock.write.assert_called_with(MessageToJson(msg).replace("\n", "")+"\n")


    @patch('environment.messages_cache.pickle')
    def test_log_normal_obj(self, pickle_mock:Mock):
        msg = CustomerPredictions(name="jack", predictions=np.arange(12), first_ts=1)
        caches.log_message(signals.COMP_USAGE_EST, msg)
        pickle_mock.dump.assert_called_with(msg, caches.file_handlers[signals.COMP_USAGE_EST])

    def test_log_unlog(self):
        msg = CustomerPredictions(name="jack", predictions=np.arange(24), first_ts=1)
        caches.log_message(signals.COMP_USAGE_EST, msg)
        msg.first_ts = 2
        caches.log_message(signals.COMP_USAGE_EST, msg)
        msg.first_ts = 3
        caches.log_message(signals.COMP_USAGE_EST, msg)

        caches._close_all_handlers()

        handler = open(os.path.join("data/testing/", "{}.pickle".format(signals.COMP_USAGE_EST)), 'rb')
        for i in range(3):
            unpickled = pickle.load(handler)
            assert unpickled.first_ts == i+1

