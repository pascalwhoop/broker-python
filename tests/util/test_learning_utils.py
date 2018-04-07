from unittest import TestCase
from unittest.mock import MagicMock

from util.learning_utils import TbWriterHelper


class TestLearningUtils(TestCase):
    def test_write_train_loss(self):
        helper = TbWriterHelper("modelname")
        helper.train_writer.add_summary = MagicMock()
        helper.write_train_loss(123)
        helper.train_writer.add_summary.assert_called_once()