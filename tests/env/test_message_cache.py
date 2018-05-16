import unittest

from pydispatch import dispatcher

from communication.grpc_messages_pb2 import PBTimeslotComplete
from communication.pubsub.grpc_adapter import publish_grpc_message
import env.messages_cache as caches

class TestMessageCache(unittest.TestCase):

    def test_cache_all_pb_messages(self):
        msg = PBTimeslotComplete(timeslotIndex=2)
        publish_grpc_message(msg)
        self.assertTrue(len(caches.PBTimeslotCompleteCache) > 0)

