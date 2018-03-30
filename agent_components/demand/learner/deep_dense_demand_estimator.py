import time
from keras import Sequential
from keras.callbacks import TerminateOnNaN, TensorBoard
from keras.layers import Dense, Activation, BatchNormalization
from keras.optimizers import sgd
import logging

logging.basicConfig(level=logging.INFO)

from agent_components.demand.learner import preprocessing

#losses = []
from agent_components.demand.learner.lstm_model import callbacks
from agent_components.demand.learner.preprocessing import DATAPOINTS_PER_TS, GamesIterator, CustomSequenceDenseTraining

logging.info("making validation data")
gi = GamesIterator()
validation_generator = CustomSequenceDenseTraining(gi, 20)
validation_data_x = []
validation_data_y = []
for i in range(len(validation_generator)):
    x, y = validation_generator[i]
    validation_data_x.extend(x)
    validation_data_y.extend(y)

tb_callback = TensorBoard(log_dir='./Graph',
                         histogram_freq=1,
                         batch_size=32,
                         write_grads=True,
                         write_graph=True,
                         write_images=True)
term_callback = TerminateOnNaN()



logging.info("starting full training")
def run_full_learning():
    mdl = get_model()
    generator = CustomSequenceDenseTraining(gi, 20)

    mdl.fit_generator(generator=generator,
                      steps_per_epoch=10000,
                      epochs=5,
                      verbose=1,
                      #validation_data=(validation_data_x, validation_data_y),
                      #callbacks=[term_callback],
                      workers=8,
                      use_multiprocessing=True,
                      shuffle=True,
                      initial_epoch=0
                      )
    #mdl.save_weights("data/consume/game{}.HDF5".format(g_number))


# input to NN --> Datapoints + 4 snapshots of previous usage (-1h, -6h, -24h, -168h)
INPUT_UNITS = DATAPOINTS_PER_TS + 4

def get_model():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(INPUT_UNITS,),))
    # input layer
    model.add(Dense(units = INPUT_UNITS))
    model.add(Dense(units=1))
    model.add(Activation('linear'))

    start = time.time()
    optimizr = sgd(lr=0.0)
    model.compile(loss='mse', optimizer=optimizr)
    logging.info('compilation time : ', time.time() - start)
    return model

run_full_learning()
