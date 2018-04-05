import time
from keras import Sequential
from keras.callbacks import TerminateOnNaN, TensorBoard
from keras.layers import Dense, Activation, BatchNormalization
from keras.optimizers import sgd
import logging

from agent_components.demand.learner.dense.dense_preprocessing import CustomSequenceDenseTraining, GamesIterator

logging.basicConfig(level=logging.INFO)

#losses = []

BATCH_SIZE        = 1  # number of sequences to feed to the model at once and whose errors are added up before propagated
SAMPLING_RATE     = 3   # assuming correlation between hours somewhere in this range (6h ago, 12h ago, 18h ago, 24h ago,..)
SEQUENCE_LENGTH   = 168 # one week sequences because that's a probable range for patterns
DATAPOINTS_PER_TS = 17  # number of datapoints in each timestep. That's customer data, weather, usage etc
VALIDATION_PART   = 0.05


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
