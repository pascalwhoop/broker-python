import numpy as np
from unittest import TestCase
from unittest.mock import Mock

from agent_components.wholesale.learning.tensorforce import create_spec
from agent_components.wholesale.learning.postprocessor import discrete_translator


class TestTensorforce(TestCase):
    def test_create_spec(self):
        spec = create_spec("discrete", "naf", "cnn_dqn_network")
        assert 'network' in spec
        assert 'states' in spec
        assert 'actions' in spec
        assert 'type' in spec

    def test_discrete_translator(self):
        env = Mock()
        #prediction
        action = np.array([5,10])
        env.get_last_known_market_price.return_value = 10
        env.purchases = []
        env.predictions = [10]
        real_action = discrete_translator(env,action)
        assert real_action[0] == -10
        assert real_action[1] == 20

        action = np.array([0,0])
        real_action = discrete_translator(env,action)
        assert real_action[0] == 0
        assert real_action[1] == 0

        action = np.array([10,10])
        real_action = discrete_translator(env,action)
        assert real_action[0] == -20
        assert real_action[1] == 20


        env.predictions = [-10]
        action = np.array([10,10])
        real_action = discrete_translator(env,action)
        assert real_action[0] == 20
        assert real_action[1] == -20
