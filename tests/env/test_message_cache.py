import unittest

from communication.grpc_messages_pb2 import PBTimeslotComplete
from communication.pubsub.grpc_adapter import publish_pb_message
import environment.messages_cache as caches

class TestMessageCache(unittest.TestCase):

    def test_cache_all_pb_messages(self):
        msg = PBTimeslotComplete(timeslotIndex=2)
        publish_pb_message(msg)
        self.assertTrue(len(caches.PBTimeslotCompleteCache) > 0)

