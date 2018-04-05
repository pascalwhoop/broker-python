import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import Sequence
from sklearn import preprocessing

from agent_components.demand.learner.gru.gru_model import SEQUENCE_LENGTH, SAMPLING_RATE, BATCH_SIZE, DATAPOINTS_PER_TS


def preprocess_x(x, y):
    _data                 = np.array(x)
    assert len(_data[0]) == 17
    hot_enc               = preprocessing.OneHotEncoder(sparse=False).fit_transform(_data[:,10:12])
    y[-1] = 0
    return np.concatenate((_data[:,0:10],
                           _data[:, 12:],
                           hot_enc,
                         #  y.reshape(-1, 1)
                           ),
                          axis=1)


def preprocess_y(y):
    return preprocessing.RobustScaler().fit_transform(y.reshape((-1,1))).flatten()

def make_customer_group_batches(game):
    """
    Generate batches across all customers with one batch per timestep
    :param game:
    :return:
    """
    x_cust = np.array(game[0]) # list of customers timeslots
    y_cust = np.array(game[1])

    #scaling the y's
    y_cust = [preprocess_y(y) for y in y_cust]
    #adding y to x (except current) and onehotting the ToD, DoW
    x_cust = [preprocess_x(x_cust[i], y_cust[i]) for i in range(len(x_cust))]

    #unwrapping customer sequences
    cust_sequences = [unwrapped_sequences(x_cust[i], y_cust[i]) for i in range(len(x_cust))]
    x_sequences = [c[0] for c in cust_sequences] #all customers x values as sequences of features
    y_sequences = [c[1] for c in cust_sequences] #all customers y values as sequences of targets

    #i want batches with ~200 per batch (one customer each)
    x_sequences = np.array(x_sequences)
    y_sequences = np.array(y_sequences)
    x_shape = x_sequences.shape
    x_sequences = x_sequences.reshape((x_shape[1],x_shape[0], int(SEQUENCE_LENGTH/SAMPLING_RATE), DATAPOINTS_PER_TS))
    y_sequences = y_sequences.reshape((x_shape[1],x_shape[0]))
    return x_sequences, y_sequences



def unwrapped_sequences(x, y):

    seq = get_generator_for_customer(x, y)
    x_cust_seq, y_cust_seq = convert_sequence_to_set(seq)
    return x_cust_seq, y_cust_seq


def convert_sequence_to_set(sequence: Sequence):
    #this is a set of tuples. We need a tuple of sets...
    x = []
    y = []
    for i in range(sequence.length):
        batch = sequence[i]
        x.extend(batch[0])
        y.extend(batch[1])
    return np.array(x), np.array(y)



def get_generator_for_customer(data, targets):
    """[keras docs](https://keras.io/preprocessing/sequence/#timeseriesgenerator)"""
    return TimeseriesGenerator(data,
                               targets,
                               length=SEQUENCE_LENGTH,
                               sampling_rate=SAMPLING_RATE,
                               batch_size=10000)

