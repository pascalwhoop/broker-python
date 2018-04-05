import unittest

import agent_components.demand.data_generator as ddg
import env.environment as env

import statefiles.state_extractor as se


class TestDemandDataGenerator(unittest.TestCase):

    def setUp(self):
        env.reset()



    def _tick_callback(self):
        ddg.make_training_rows(env)


    def test_make_training_rows(self):
        test_file_path = "tests/test.state"
        states = se.get_states_from_file(test_file_path)
        se.parse_state_lines(states, self._tick_callback)

        for producer in ddg.consume_data.values():
            self.assertEqual(17, len(producer[0][0]))
            self.assertEqual(4,  len(producer[0]))
            self.assertEqual(4,  len(producer[1]))
            self.assertEqual(2, len(producer))
