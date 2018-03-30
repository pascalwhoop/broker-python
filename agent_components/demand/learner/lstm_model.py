from keras.layers.core import Dense, Activation, Dropout
from keras.layers import CuDNNLSTM
from keras.callbacks import TensorBoard, TerminateOnNaN
from keras.models import Sequential
from keras.optimizers import rmsprop
import time

from keras.utils import Sequence

from agent_components.demand.learner.preprocessing import BATCH_SIZE, SEQUENCE_LENGTH, SAMPLING_RATE, DATAPOINTS_PER_TS




callbacks = []
callbacks.append(TensorBoard(log_dir='./Graph',
                         histogram_freq=1,
                         batch_size=32,
                         write_grads=True,
                         write_graph=True,
                         write_images=True))
callbacks.append(TerminateOnNaN())

def get_model():
    model = Sequential()

    #input layer
    model.add(CuDNNLSTM(input_shape=(SEQUENCE_LENGTH/SAMPLING_RATE, DATAPOINTS_PER_TS),
                        units=100,
                        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(CuDNNLSTM(units=200,
                        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(CuDNNLSTM(units=200,
                        return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.add(Activation('linear'))

    start = time.time()
    optimizr = rmsprop(lr=0.0001)
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
