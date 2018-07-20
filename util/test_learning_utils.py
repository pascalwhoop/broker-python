from unittest import TestCase
from unittest.mock import MagicMock, patch, mock_open

from sklearn.preprocessing import MinMaxScaler

from util.learning_utils import TbWriterHelper
import numpy as np


#class TestLearningUtils(TestCase):
#    def test_write_train_loss(self):
#        helper = TbWriterHelper("test_modelname")
#        helper.write_train_loss(123)
#        #assume no error
