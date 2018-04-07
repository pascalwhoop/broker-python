import os

from agent_components.demand.learning.lstm import lstm_preprocessing as preprocessing
from agent_components.demand.learning.lstm import lstm_model
from agent_components.demand.learning.lstm.lstm_model import MODEL_NAME

mdl = lstm_model.get_model()
#losses = []

def run_full_learning():
    # iterate over games (holding customer lists)
    for g_number, game in enumerate(preprocessing.GamesIterator()):
        # iterating over customers in game
        # taking all - 10 sequences of each customer
        for sequences in preprocessing.LSTMCustomerIterator(game[0], game[1]):
            lstm_model.fit_with_generator(mdl, sequences[0], sequences[1])
        # evaluating loss on last 10 timesteps of each customer
        #for customer_sequence in preprocessing.CustomerIterator(game[0][-10:],game[1][-10:]):
        #    loss = mdl.evaluate_generator(customer_sequence, workers=8,use_multiprocessing=True)
        #    print(loss)
        #    losses.append(loss)
        dir = "data/consume_models/{}/".format(MODEL_NAME)
        os.makedirs(dir, exist_ok=True)
        mdl.save_weights(os.path.join(dir, "game{}.HDF5".format(g_number)))

run_full_learning()










# ### Clean pipeline try number 2
# 
# Alright. What's needed is a pipeline as such:
# 
# 1. iterate over games
#     1. Iterate over customers
#         - train on customer N times
#         - generate sequences from customer using [TimeseriesGenerator](https://keras.io/preprocessing/sequence/#timeseriesgenerator)
#     2. Evaluate Performance on test set
# 
# 
# For the sequences, a function is needed that generates a Sequence from a customers Timeseries Data
#         
#     
