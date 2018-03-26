import agent_components.demand.data_generator as dg



training_data = []
labels        = []
def store_game(game_data):
    for c in game_data.values():
        training_data.append(c[0])
        labels.append(c[1])



def round_callback():
    print("round passed")
    store_game(dg.consume_data[dg.game_counter])
    dg.round_callback()


def run():
    dg.se.run_through_all_files(dg.tick_callback, round_callback)


#import profile
#profile.run('run()')
run()
