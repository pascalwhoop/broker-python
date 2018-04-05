import os
import numpy as np
import tensorflow as tf
from agent_components.demand.learner.gru import gru_model, gru_preprocessing
from agent_components.demand.learner.gru.gru_across_customers_preprocessing import make_customer_group_batches
from agent_components.demand.learner.gru.gru_model import MODEL_NAME, callbacks

mdl = gru_model.get_model()
#losses = []

summaries_dir = "Graph/{}".format(MODEL_NAME)
train_writer = tf.summary.FileWriter(summaries_dir + '/train')
test_writer = tf.summary.FileWriter(summaries_dir + '/test')


def run_full_learning():
    # iterate over games (holding customer lists)
    for g_number, game in enumerate(gru_preprocessing.GamesIterator()):
        # iterating over customers in game
        batches_x, batches_y = make_customer_group_batches(game)
        #iterating over all but last batch
        losses = []
        prev_batch_y = None
        stupid_guess_loss_sum = 0
        for i in range(len(batches_x)-1):
           loss =  mdl.train_on_batch(batches_x[i], batches_y[i])
           losses.append(loss)
           stupid_guess_loss = 0
           if prev_batch_y is not None:
              stupid_guess_loss = ((prev_batch_y - batches_y[i])** 2).mean()
              stupid_guess_loss_sum = stupid_guess_loss_sum + (1/(i+1)*(stupid_guess_loss - stupid_guess_loss_sum))
           print("loss now {0:.2f} -- avg 10 {1:.2f} -- avg all {2:.2f} -- stupid {3:.2f} -- stup sum {4:.2f}"
                 .format(loss,
                         np.average(np.array(losses[-10:])),
                         np.average(np.array(losses)),
                         stupid_guess_loss,
                         stupid_guess_loss_sum))
           prev_batch_y = batches_y[i]
           #train_writer.add_summary(loss)
        #on last batch, test
        loss = mdl.test_on_batch(batches_x[-1], batches_y[-1])
        print("test {}".format(loss))
        #test_writer.add_summary(loss)

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
