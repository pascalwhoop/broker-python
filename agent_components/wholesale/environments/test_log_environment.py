from collections import deque

import numpy as np


class MagickMock(object):
    pass


def make_mock_demand_data():
    return list(range(1, 50))


def make_mock_active_timeslots(data):
    # mock the active timesteps
    return deque([row[0] for row in data][:24], maxlen=24)


def make_mock_wholesale_data():
    # creating wholesale style mock data
    wholesale_data_header = np.zeros((50, 3), dtype=np.int32)
    wholesale_data_header[:, 0] = np.arange(363, 363 + 50).transpose()

    data_core = np.zeros((50, 24, 2), dtype=np.float32)
    # iterate over the rows
    for i in range(len(data_core)):
        # and each market clearing for each of the 24 times the ts was traded
        for j in range(len(data_core[i])):
            # mwh to full numbers
            data_core[i][j][0] = i + 1
            # price to 1/10th that
            data_core[i][j][1] = (i + 1) / 10

    wholesale_data = []
    for i in range(50):
        row = []
        row.extend(wholesale_data_header[i])
        row.extend(list(data_core[i]))
        wholesale_data.append(row)
    return wholesale_data


def make_mock_averages():
    # same as wholesale_data, 50 entries with averages being 1/10th the index+1
    return [i / 10 for i in range(1, 51)]
    pass





