import unittest
from unittest.mock import Mock

import agent_components.demand.generate_data.data_generator as l
from model.customer_info import CustomerInfo
from model.rate import Rate
from model.tariff_transaction import TariffTransaction
from util.bunch import Bunch
from env import environment
env_ = environment.get_instance()


class TestDemandDataGenerator(unittest.TestCase):

    def test_ensure_customer(self):
        game = {}
        customer = CustomerInfo(id_="chicken")
        l._ensure_customer(game, customer)
        self.assertEqual(2, len(game[customer.id_]))

    def test_add_consume_data(self):
        row = []
        transactions = [TariffTransaction(kWh=1), TariffTransaction(kWh=3)]
        l.add_consume_data(row, transactions)
        self.assertEqual(1, len(row))
        self.assertEqual(4, row[0])

    def test_add_rate_data(self):
        transactions = [TariffTransaction(txType="CONSUME", tariffSpec="123")]

        mock_grfct = Mock(return_value=Rate(id_="123"))
        tariff_store_mock = Bunch(get_rate_for_customer_transactions=mock_grfct)
        env_.tariff_store = tariff_store_mock

        row = []
        l.add_rate_data(env_, row, transactions)
        self.assertEqual(5, len(row))

    counter = 0

    def _tick_callback(self):
        self.counter +=1
        l.make_training_rows(l.env)
        if self.counter > 5:
            game0 = l.consume_data[0]
            self.assertEqual(200, len(game0))
            self.assertEqual(4, len(game0.values()[0]))
            raise WindowsError


#       def test_generate_onehundred(self):
#        l.se.run_through_all_files(self._tick_callback, l.round_callback)
