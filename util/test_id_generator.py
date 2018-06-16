import unittest

import util.id_generator as idg

line = '<broker-accept prefix="2" key="dlop5b" serverTime="1521046957413"/>'


class TestIdGenerator(unittest.TestCase):

    def test_handle_broker_accept_line(self):
        idg.handle_broker_accept_line(line)
        self.assertEqual(idg.key, "dlop5b")
        self.assertEqual(idg.prefix, "2")
