import unittest
from unittest.mock import patch

from communication.grpc_messages_pb2 import PBTimeslotComplete
from communication.pubsub import signals
import environment.messages_cache as caches


class TestMessageCache(unittest.TestCase):

    def test_cache_all_pb_messages(self):
        msg = PBTimeslotComplete(timeslotIndex=2)
        caches.store_message(None, signals.PB_TIMESLOT_COMPLETE, msg)
        self.assertTrue(len(caches.PBTimeslotCompleteCache) > 0)

