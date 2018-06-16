import unittest

import numpy as np

import util.config as cfg
from agent_components.demand.learning.preprocessing import DemandCustomerSequence




class TestDemandPreprocessing(unittest.TestCase):
    # def test_DemandCustomerSequenceIterator(self):
    #    iter = DemandCustomerSequenceIterator(fake_y)
    #    for seq in iter:
    #        print(seq)
    #        self.assertFalse(True)
    def setUp(self):
        self.instance = DemandCustomerSequence(3, cust_ts_x, cust_targets)
        self.assertEqual((100, 200, cfg.DEMAND_DATAPOINTS_PER_TS-1), np.array(self.instance.cust_features).shape)

    def test_DemandCustomerSequence_shape(self):
        self.assertEqual(200, len(self.instance))

        batch_of_sequences = self.instance.__getitem__(0)
        self.assertEqual((100, 48, cfg.DEMAND_DATAPOINTS_PER_TS), batch_of_sequences[0].shape)
        self.assertEqual((100), batch_of_sequences[1].shape[0])

        # no offset at beginning anymore
        batch_of_sequences = self.instance.__getitem__(60)
        self.assertEqual((100, 48, cfg.DEMAND_DATAPOINTS_PER_TS), batch_of_sequences[0].shape)

    def test_DemandCustomerSequence_beginning_padding(self):
        # testing for both "demand delay" (3 above) and instance 50
        # expecting to get a 48 ts long sequence with first row with 0s
        x, y = self.instance.__getitem__(50)
        self.assertEqual(0, np.sum(x[:, 0, :]))


    def test_DemandCustomerSequence_only_usage(self):
        instance = DemandCustomerSequence(3, None, cust_targets)
        x, y = instance.__getitem__(60)
        self.assertEqual((100, 48, 1), x.shape)

    def test_DemandCustomerSequence_flatten_seq(self):
        #tests the flatten_sequences flag for the generator
        instance = DemandCustomerSequence(3, None, cust_targets, flatten_sequences=True)
        x, y, = instance.__getitem__(60)
        self.assertEqual((100,48), x.shape)

    def test_DemandCustomerSequence_end_padding(self):
        """We assume that a forecast distance of 3 means that the last 2 timeslots in a sequence are not known. That is because
        with a sequence length of e.g. 48, the first 24 sequence samples have entries, but the last 24 dont with a forecast
        distance of 24. The algorithm doens't know the next 24 timeslots yet but must still make predictions"""
        ins = self.instance
        x, y = ins.__getitem__(60)
        self.assertEqual(0, x[:, -2].sum())

    def test_DemandCustomerSequence_from(self):
        from_ = self.instance._calculate_from(50)
        self.assertEqual(2, from_)
        from_ = self.instance._calculate_from(60)
        self.assertEqual(12, from_)

    def test_DemandCustomerSequence_till(self):
        till_ = self.instance._calculate_till(50, 1)
        self.assertEqual(47, till_)
        till_ = self.instance._calculate_till(60, 0)
        self.assertEqual(58, till_)

    def test_DemandCustomerSequence_offset(self):
        ins = self.instance
        base_offset = cfg.DEMAND_SEQUENCE_LENGTH - 30
        offset = ins._calculate_offset(30)
        self.assertEqual(base_offset + ins.forecast_distance, offset)

    def test_DemandCustomerSequence_with_other_sequence_length(self):
        cfg.DEMAND_SEQUENCE_LENGTH = 100
        x, y = self.instance.__getitem__(99)
        self.assertEqual((100, 100, cfg.DEMAND_DATAPOINTS_PER_TS), x.shape)
        self.assertEqual(0, self.instance._calculate_from(60))
        self.assertEqual(20, self.instance._calculate_from(120))

        # let's reset the cfg changes made
        import importlib
        importlib.reload(cfg)  # reloading own helper often, it's still changing during writing


values = [[10, 1, 0.0, 0.0, 0.0, -0.5, 0.0, 0, 0.0, 0.0, 2, 7, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.5, 0.0, 0, 0.0, 0.0, 0, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.5, 0.0, 0, 0.0, 0.0, 0, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.058, 0.0, 0, 0.0, 0.0, 1, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.058, 0.0, 0, 0.0, 0.0, 2, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.058, 0.0, 0, 0.0, 0.0, 3, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.058, 0.0, 0, 0.0, 0.0, 4, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.058, 0.0, 0, 0.0, 0.0, 5, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.058, 0.0, 0, 0.0, 0.0, 6, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 7, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 8, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 9, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 10, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 11, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 12, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 13, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 14, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 15, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 16, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 17, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 18, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 19, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 20, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 21, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 22, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 23, 1, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 0, 2, 24, 14.0, 17.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 1, 2, 24, 13.0, 15.0, 0.0, 0.19],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 2, 2, 24, 13.0, 14.0, 0.0, 0.14],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 3, 2, 24, 12.0, 13.0, 0.0, 0.09],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 4, 2, 24, 11.0, 14.0, 0.0, 0.28],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 5, 2, 24, 11.0, 14.0, 0.0, 0.27],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 6, 2, 24, 11.0, 15.0, 0.0, 0.26],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 7, 2, 24, 14.0, 17.0, 0.0, 0.31],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 8, 2, 24, 17.0, 19.0, 0.0, 0.37],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 9, 2, 24, 19.0, 21.0, 0.0, 0.42],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 10, 2, 24, 21.0, 22.0, 0.0, 0.45],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 11, 2, 24, 23.0, 23.0, 0.0, 0.49],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 12, 2, 24, 24.0, 24.0, 0.0, 0.52],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 13, 2, 24, 25.0, 24.0, 0.0, 0.53],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 14, 2, 24, 25.0, 22.0, 0.0, 0.58],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 15, 2, 24, 26.0, 22.0, 0.0, 0.6],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 16, 2, 24, 26.0, 22.0, 0.0, 0.57],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 17, 2, 24, 25.0, 22.0, 0.0, 0.55],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 18, 2, 24, 25.0, 22.0, 0.0, 0.52],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 19, 2, 24, 23.0, 22.0, 0.0, 0.45],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 20, 2, 24, 21.0, 22.0, 0.0, 0.38],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 21, 2, 24, 19.0, 22.0, 0.0, 0.31],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 22, 2, 24, 17.0, 22.0, 0.0, 0.23],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 23, 2, 24, 16.0, 22.0, 0.0, 0.15],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 0, 3, 24, 14.0, 22.0, 0.0, 0.07],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 1, 3, 24, 14.0, 21.0, 0.0, 0.09],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 2, 3, 24, 13.0, 20.0, 0.0, 0.11],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 3, 3, 24, 13.0, 18.0, 0.0, 0.13],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 4, 3, 24, 12.0, 17.0, 0.0, 0.16],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 5, 3, 24, 12.0, 16.0, 0.0, 0.13],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 6, 3, 24, 11.0, 15.0, 0.0, 0.1],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 7, 3, 24, 15.0, 15.0, 0.0, 0.1],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 8, 3, 24, 19.0, 16.0, 0.0, 0.1],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 9, 3, 24, 22.0, 17.0, 0.0, 0.1],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 10, 3, 24, 24.0, 17.0, 0.0, 0.1],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 11, 3, 24, 26.0, 18.0, 0.0, 0.1],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 12, 3, 24, 28.0, 18.0, 0.0, 0.1],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 13, 3, 24, 28.0, 19.0, 0.0, 0.1],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 14, 3, 24, 29.0, 20.0, 0.0, 0.1],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 15, 3, 24, 31.0, 31.0, 0.0, 0.1],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 16, 3, 24, 31.0, 31.0, 0.0, 0.18],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 17, 3, 24, 30.0, 30.0, 0.0, 0.21],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 18, 3, 24, 29.0, 30.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 19, 3, 24, 27.0, 28.0, 0.0, 0.29],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 20, 3, 24, 24.0, 26.0, 0.0, 0.34],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 21, 3, 24, 22.0, 24.0, 0.0, 0.39],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 22, 3, 24, 20.0, 21.0, 0.0, 0.3],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 23, 3, 24, 19.0, 18.0, 0.0, 0.2],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 0, 4, 24, 17.0, 15.0, 0.0, 0.11],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 1, 4, 24, 16.0, 17.0, 0.0, 0.09],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 2, 4, 24, 15.0, 19.0, 0.0, 0.08],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 3, 4, 24, 13.0, 18.0, 0.0, 0.06],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 4, 4, 24, 12.0, 20.0, 0.0, 0.05],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 5, 4, 24, 12.0, 21.0, 0.0, 0.04],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 6, 4, 24, 12.0, 22.0, 0.0, 0.03],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 7, 4, 24, 14.0, 21.0, 0.0, 0.03],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 8, 4, 24, 17.0, 20.0, 0.0, 0.03],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 9, 4, 24, 19.0, 18.0, 0.0, 0.03],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 10, 4, 24, 21.0, 17.0, 0.0, 0.14],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 11, 4, 24, 23.0, 16.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 12, 4, 24, 25.0, 15.0, 0.0, 0.35],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 13, 4, 24, 26.0, 16.0, 0.0, 0.38],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 14, 4, 24, 27.0, 17.0, 0.0, 0.4],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 15, 4, 24, 28.0, 18.0, 0.0, 0.43],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 16, 4, 24, 28.0, 19.0, 0.0, 0.41],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 17, 4, 24, 27.0, 22.0, 0.0, 0.35],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 18, 4, 24, 27.0, 21.0, 0.0, 0.31],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 19, 4, 24, 24.0, 19.0, 0.0, 0.31],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 20, 4, 24, 22.0, 18.0, 0.0, 0.31],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 21, 4, 24, 19.0, 17.0, 0.0, 0.31],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 22, 4, 24, 18.0, 15.0, 0.0, 0.26],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 23, 4, 24, 17.0, 14.0, 0.0, 0.22],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 0, 5, 24, 16.0, 13.0, 0.0, 0.17],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 1, 5, 24, 15.0, 11.0, 0.0, 0.19],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 2, 5, 24, 14.0, 9.0, 0.0, 0.2],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 3, 5, 24, 13.0, 8.0, 0.0, 0.22],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 4, 5, 24, 12.0, 8.0, 0.0, 0.21],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 5, 5, 24, 13.0, 11.0, 0.0, 0.13],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 6, 5, 24, 12.0, 11.0, 0.0, 0.09],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 7, 5, 24, 16.0, 11.0, 0.0, 0.11],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 8, 5, 24, 19.0, 11.0, 0.0, 0.12],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 9, 5, 24, 23.0, 11.0, 0.0, 0.14],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 10, 5, 24, 24.0, 12.0, 0.0, 0.19],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 11, 5, 24, 26.0, 12.0, 0.0, 0.23],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 12, 5, 24, 27.0, 13.0, 0.0, 0.28],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 13, 5, 24, 28.0, 15.0, 0.0, 0.31],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 14, 5, 24, 29.0, 17.0, 0.0, 0.33],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 15, 5, 24, 29.0, 17.0, 0.0, 0.36],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 16, 5, 24, 29.0, 17.0, 0.0, 0.44],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 17, 5, 24, 29.0, 17.0, 0.0, 0.42],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 18, 5, 24, 28.0, 17.0, 0.0, 0.39],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 19, 5, 24, 26.0, 15.0, 0.0, 0.37],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 20, 5, 24, 24.0, 13.0, 0.0, 0.36],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 21, 5, 24, 21.0, 11.0, 0.0, 0.34],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 22, 5, 24, 20.0, 12.0, 0.0, 0.31],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 23, 5, 24, 18.0, 12.0, 0.0, 0.29],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 0, 6, 24, 17.0, 13.0, 0.0, 0.26],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 1, 6, 24, 16.0, 12.0, 0.0, 0.24],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 2, 6, 24, 14.0, 11.0, 0.0, 0.22],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 3, 6, 24, 13.0, 9.0, 0.0, 0.2],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 4, 6, 24, 13.0, 8.0, 0.0, 0.25],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 5, 6, 24, 13.0, 9.0, 0.0, 0.29],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 6, 6, 24, 13.0, 9.0, 0.0, 0.34],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 7, 6, 24, 16.0, 10.0, 0.0, 0.34],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 8, 6, 24, 19.0, 11.0, 0.0, 0.34],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 9, 6, 24, 22.0, 11.0, 0.0, 0.34],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 10, 6, 24, 24.0, 13.0, 0.0, 0.38],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 11, 6, 24, 26.0, 15.0, 0.0, 0.43],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 12, 6, 24, 27.0, 17.0, 0.0, 0.47],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 13, 6, 24, 28.0, 18.0, 0.0, 0.49],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 14, 6, 24, 28.0, 19.0, 0.0, 0.5],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 15, 6, 24, 28.0, 15.0, 0.0, 0.52],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 16, 6, 24, 27.0, 14.0, 0.0, 0.54],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 17, 6, 24, 27.0, 12.0, 0.0, 0.55],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 18, 6, 24, 27.0, 11.0, 0.0, 0.57],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 19, 6, 24, 24.0, 13.0, 0.0, 0.57],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 20, 6, 24, 22.0, 15.0, 0.0, 0.57],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 21, 6, 24, 19.0, 17.0, 0.0, 0.57],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 22, 6, 24, 18.0, 16.0, 0.0, 0.44],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 23, 6, 24, 16.0, 15.0, 0.0, 0.32],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 0, 7, 24, 15.0, 15.0, 0.0, 0.19],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 1, 7, 24, 14.0, 14.0, 0.0, 0.19],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 2, 7, 24, 13.0, 12.0, 0.0, 0.19],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 3, 7, 24, 12.0, 11.0, 0.0, 0.19],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 4, 7, 24, 11.0, 15.0, 0.0, 0.18],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 5, 7, 24, 12.0, 15.0, 0.0, 0.19],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 6, 7, 24, 12.0, 15.0, 0.0, 0.19],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 7, 7, 24, 15.0, 13.0, 0.0, 0.21],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 8, 7, 24, 18.0, 11.0, 0.0, 0.23],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 9, 7, 24, 21.0, 9.0, 0.0, 0.25],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 10, 7, 24, 22.0, 12.0, 0.0, 0.34],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 11, 7, 24, 24.0, 14.0, 0.0, 0.43],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 12, 7, 24, 26.0, 17.0, 0.0, 0.52],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 13, 7, 24, 26.0, 19.0, 0.0, 0.53],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 14, 7, 24, 27.0, 22.0, 0.0, 0.54],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 15, 7, 24, 27.0, 24.0, 0.0, 0.55],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 16, 7, 24, 27.0, 22.0, 0.0, 0.47],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 17, 7, 24, 27.0, 20.0, 0.0, 0.38],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 18, 7, 24, 24.0, 24.0, 0.0, 0.3],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 19, 7, 24, 23.0, 21.0, 0.0, 0.3],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 20, 7, 24, 22.0, 18.0, 0.0, 0.3],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 21, 7, 24, 21.0, 15.0, 0.0, 0.3],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 22, 7, 24, 19.0, 15.0, 0.0, 0.25],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 23, 7, 24, 17.0, 15.0, 0.0, 0.2],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 0, 1, 24, 16.0, 15.0, 0.0, 0.15],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 1, 1, 24, 15.0, 14.0, 0.0, 0.15],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 2, 1, 24, 15.0, 12.0, 0.0, 0.15],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 3, 1, 24, 14.0, 11.0, 0.0, 0.15],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 4, 1, 24, 13.0, 11.0, 0.0, 0.19],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 5, 1, 24, 13.0, 12.0, 0.0, 0.29],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 6, 1, 24, 13.0, 13.0, 0.0, 0.26],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 7, 1, 24, 16.0, 14.0, 0.0, 0.26],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 8, 1, 24, 19.0, 15.0, 0.0, 0.26],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 9, 1, 24, 22.0, 17.0, 0.0, 0.26],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 10, 1, 24, 24.0, 18.0, 0.0, 0.34],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 11, 1, 24, 26.0, 19.0, 0.0, 0.41],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 12, 1, 24, 27.0, 21.0, 0.0, 0.49],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 13, 1, 24, 28.0, 22.0, 0.0, 0.5],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 14, 1, 24, 29.0, 23.0, 0.0, 0.51],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 15, 1, 24, 29.0, 24.0, 0.0, 0.52],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 16, 1, 24, 29.0, 24.0, 0.0, 0.5],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 17, 1, 24, 28.0, 23.0, 0.0, 0.47],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 18, 1, 24, 28.0, 22.0, 0.0, 0.45],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 19, 1, 24, 26.0, 21.0, 0.0, 0.42],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 20, 1, 24, 23.0, 20.0, 0.0, 0.4],
          [10, 1, 0.0, 0.0, 0.0, -0.064, 0.0, 0, 0.0, 0.0, 21, 1, 24, 21.0, 18.0, 0.0, 0.37],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 22, 1, 24, 20.0, 20.0, 0.0, 0.35],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 23, 1, 24, 19.0, 21.0, 0.0, 0.32],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 0, 2, 24, 17.0, 22.0, 0.0, 0.3],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 1, 2, 24, 16.0, 22.0, 0.0, 0.3],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 2, 2, 24, 15.0, 22.0, 0.0, 0.3],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 3, 2, 24, 14.0, 22.0, 0.0, 0.3],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 4, 2, 24, 14.0, 22.0, 0.0, 0.36],
          [10, 1, 0.0, 0.0, 0.0, -0.044, 0.0, 0, 0.0, 0.0, 5, 2, 24, 14.0, 22.0, 0.0, 0.33]]
cust_targets = [range(200)] * 100
cust_ts_x = [values] * 100