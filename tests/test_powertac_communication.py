import unittest
from unittest.mock import MagicMock
import util.powertac_communication as comm
import util.id_generator as idg
import tacgrpc.grpc_pb2 as model

class TestPowertacCommunication(unittest.TestCase):

    xml_ = '<broker-accept prefix="2" key="dlop5b" serverTime="1521046957413"/>'
    def test_broker_accept_autoremoval(self):
        accept = model.XmlMessage(counter= 1, rawMessage=self.xml_)

        comm._message_stub.registerListener = MagicMock(return_value=[accept])
        idg.handle_broker_accept_line = MagicMock()
        # exec
        self.assertEqual(1, len(comm.interceptors))
        comm._connect_incoming()
        self.assertEqual(0, len(comm.interceptors))

    def test_prefixing(self):
        idg.key = "chicken"
        comm.put(self.xml_)
        g_xml = comm._out_queue.get()
        self.assertEqual("chicken"+self.xml_, g_xml.rawMessage)
