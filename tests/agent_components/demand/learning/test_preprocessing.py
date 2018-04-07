import unittest

import numpy as np

import util.config as cfg
from agent_components.demand.learning.preprocessing import DemandCustomerSequence

cust_targets = [range(200)] * 100
features = np.array(range(17)) + 1
timesteps = [features] * 200
cust_ts_x = [timesteps] * 100


class TestDemandPreprocessing(unittest.TestCase):
    # def test_DemandCustomerSequenceIterator(self):
    #    iter = DemandCustomerSequenceIterator(fake_y)
    #    for seq in iter:
    #        print(seq)
    #        self.assertFalse(True)
    def setUp(self):
        self.instance = DemandCustomerSequence(3, cust_targets, cust_ts_x)
        self.assertEqual((100, 200, 17), np.array(self.instance.cust_features).shape)

    def test_DemandCustomerSequence_shape(self):
        self.assertEqual(200, len(self.instance))

        batch_of_sequences = self.instance.__getitem__(0)
        self.assertEqual((100, 48, 18), batch_of_sequences[0].shape)
        self.assertEqual((100), batch_of_sequences[1].shape[0])

        # no offset at beginning anymore
        batch_of_sequences = self.instance.__getitem__(60)
        self.assertEqual((100, 48, 18), batch_of_sequences[0].shape)

    def test_DemandCustomerSequence_beginning_padding(self):
        # testing for both "demand delay" (3 above) and instance 50
        # expecting to get a 48 ts long sequence with first row with 0s
        x, y = self.instance.__getitem__(50)
        self.assertEqual(0, np.sum(x[:,0,:]))


    def test_DemandCustomerSequence_end_padding(self):
        """We assume that a forecast distance of 3 means that the last 2 timeslots in a sequence are not known. That is because
        with a sequence length of e.g. 48, the first 24 sequence samples have entries, but the last 24 dont with a forecast
        distance of 24. The algorithm doens't know the next 24 timeslots yet but must still make predictions"""
        ins = self.instance
        x, y = ins.__getitem__(60)
        self.assertEqual(0, x[:,-2].sum())

    def test_DemandCustomerSequence_from(self):
        from_ = self.instance.calculate_from(50)
        self.assertEqual(2, from_)
        from_ = self.instance.calculate_from(60)
        self.assertEqual(12, from_)
    def test_DemandCustomerSequence_till(self):
        till_ = self.instance.calculate_till(50, 1)
        self.assertEqual(47, till_)
        till_ = self.instance.calculate_till(60, 0)
        self.assertEqual(58, till_)
    def test_DemandCustomerSequence_offset(self):
        ins = self.instance
        base_offset = cfg.DEMAND_SEQUENCE_LENGTH - 30
        offset = ins.calculate_offset(30)
        self.assertEqual(base_offset+ins.forecast_distance, offset)

    def test_DemandCustomerSequence_with_other_sequence_length(self):
        cfg.DEMAND_SEQUENCE_LENGTH = 100
        x,y = self.instance.__getitem__(99)
        self.assertEqual((100, 100, 18), x.shape)
        self.assertEqual(0, self.instance.calculate_from(60))
        self.assertEqual(20, self.instance.calculate_from(120))

        #let's reset the cfg changes made
        import importlib
        importlib.reload(cfg)  # reloading own helper often, it's still changing during writing
