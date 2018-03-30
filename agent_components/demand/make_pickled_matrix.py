import os

import agent_components.demand.data_generator as dg
import pickle

training_data = []
labels_data   = []

def pickel_data():
    global training_data, labels_data

    path               = "data/consume/"
    training_file_name = "training.pickle"
    labels_file_name   = "labels.pickle"

    t_p = os.path.abspath("{}game{}{}".format(path, dg.game_counter, training_file_name))
    with open(t_p, "wb") as f:
        pickle.dump(training_data, f)

    l_f = os.path.abspath("{}game{}{}".format(path, dg.game_counter, labels_file_name))
    with open(l_f, "wb") as f:
        pickle.dump(labels_data, f)

    training_data = []
    labels_data   = []



def store_game(consume_data):
    for c in consume_data.values():
        #each customer.. one array
        c_training =c[0]
        c_labels = c[1]
        training_data.append(c[0])
        labels_data.append(c[1])


def verify_cleared():
    print("len training_data {}"             .format(len(training_data)))
    print("len labels data {}"               .format(len(labels_data)))
    print("len env values {}  {}  {}  {}  {}".format(len(dg.env.rates), len(dg.env.tariffs), len(dg.env.transactions), dg.env.current_timestep, len(dg.env.weather_reports)))


def round_callback():
    print("round passed")
    print("storing game in global list")
    store_game(dg.consume_data)

    verify_cleared()
    pickel_data()

    print("next game...")
    dg.round_callback()
    verify_cleared()


def run():
    dg.se.run_through_all_files(dg.tick_callback, round_callback)


#import profile
#profile.run('run()')
