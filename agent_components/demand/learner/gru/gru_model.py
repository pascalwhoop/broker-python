from keras.layers.core import Dense, Activation, Dropout
from keras.layers import CuDNNLSTM, BatchNormalization, CuDNNGRU, regularizers
from keras.callbacks import TensorBoard, TerminateOnNaN
from keras.models import Sequential
from keras.optimizers import rmsprop
import time

from keras.utils import Sequence


BATCH_SIZE        = 32  # TODO... higher? number of sequences to feed to the model at once and whose errors are added up before propagated
SAMPLING_RATE     = 2   # assuming correlation between hours somewhere in this range (6h ago, 12h ago, 18h ago, 24h ago,..)
SEQUENCE_LENGTH   = 48 # one week sequences because that's a probable range for patterns
#DATAPOINTS_PER_TS = 17  # number of datapoints in each timestep. That's customer data, weather, usage etc
DATAPOINTS_PER_TS = 46  # sparse version
#DATAPOINTS_PER_TS = 1  # solely based on previous usage version
VALIDATION_PART   = 0.05
MODEL_NAME        = "gru_customer_whole_batch"



callbacks = []
callbacks.append(TensorBoard(log_dir='./Graph/{}/'.format(MODEL_NAME),
                      #   histogram_freq=1,
                         batch_size=32,
                         write_grads=True,
                      #   write_graph=True,
                      #   write_images=True)
                 ))
callbacks.append(TerminateOnNaN())

def get_model():
    model = Sequential()

    #input layer
    model.add(CuDNNGRU (input_shape=(SEQUENCE_LENGTH/SAMPLING_RATE, DATAPOINTS_PER_TS),
                        units=DATAPOINTS_PER_TS,
                        kernel_regularizer=regularizers.l1(0.01),
                        return_sequences=False,
                        ))
    #model.add(CuDNNLSTM(units=200,
    #                    return_sequences=True))
    #model.add(Dropout(0.2))
    #model.add(CuDNNLSTM(units=200,
    #                    return_sequences=False))
    #model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.add(Activation('linear'))

    start = time.time()
    optimizr = rmsprop(lr=0.01)
    model.compile(loss='mse', optimizer=optimizr)
    print('compilation time : ', time.time() - start)
    return model



def fit_with_generator(model, train_generator: Sequence, validation_set):
    model.fit_generator(train_generator,
                        steps_per_epoch=None,  #a generator size (aka one customer) is an epoch
                        epochs=1,
                        verbose=2,  #progress bar, 2 = line per epoch
                        callbacks=callbacks,
                        validation_data=validation_set,
                        #validation_steps=None,
                        class_weight=None,
                        max_queue_size=10,
                        workers=8,
                        use_multiprocessing=True,
                        shuffle=True,
                        initial_epoch=0)
