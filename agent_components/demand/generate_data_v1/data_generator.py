# learning to forecast the usage patterns of a customer

# is this an LSTM? yes it's partially correlated with previous usage, not only with that of 24h past. 

# what goes in?

# customerBootstrapData | weatherForecast | past 24h usage profile | 1 week ago profile | t-x forecast distance 

# what goes out

# usage kWh

# how do I get those things? 
# iterate through state files
# save all weather forecasts / usage profiles / ...
# save usage profiles somewhere in customer? 
import logging
from functools import reduce
from typing import List

from statefiles.env import environment
from statefiles.env.environment import Environment
from model.customer_info import CustomerInfo
from model.tariff_transaction import TransactionType, TariffTransaction

# holds time series for consume data for several customers
# customer {}
#       [training[rounds][10], result[rounds[1]]]
consume_data = {}

game_counter = 0


def round_callback():
    environment.reset_instance()
    global consume_data
    consume_data = {}
    global game_counter
    tick = 0
    game_counter += 1


def _ensure_customer(game, customer: CustomerInfo):
    if customer.id_ not in game:
        # list with 2 lists. Samples and Results (what the customer used
        game[customer.id_] = [[], []]


def make_training_rows(env: Environment):
    """iterating over the current snapshot of the state of the game and getting relevant data to build samples"""
    game = consume_data

    # we take a sample for each customer at each timeslot
    for customer in env.tariff_store.customers.values():
        row = add_customer_data(customer, game)

        """if we want to learn also based on type of tariff --> add this"""
        ##getting customer tariff data
        # for t in reversed(transactions[customer.id_]):
        #    if t.txType == TransactionType.SIGNUP:
        #        tariff = env.tariffs[t.tariffSpec]

        # getting rate applicable (next few rows)
        customer_transactions = env.tariff_store.transactions[customer.id_]
        if len(customer_transactions) == 0:
            return
        transactions = [t for t in customer_transactions[-1] if
                        t.txType == TransactionType.CONSUME or TransactionType.PRODUCE]

        add_rate_data(env, row, transactions)

        # more metadata (tod, dow, weather)
        add_time_data(env, row)
        add_weather_data(env, row)

        # getting consume of today
        add_consume_data(row, transactions)

        # let's  check if this is worth adding. if so, we add the input to list0 and the result to list1
        # [0,4]  customer_metadata
        # [5,9]  rate_metadata
        # [10,11]time
        # [13,17]weather
        # [18]   consume_row
        if len(row) is 18:
            game[customer.id_][0].append(row[0:-1])
            game[customer.id_][1].append(row[-1])
        else:
            logging.warning("wrong length {}".format(len(row)))
            logging.warning(row)


def add_consume_data(row, transactions: List[TariffTransaction]):
    kWh = reduce(lambda sum, i: sum + i.kWh, transactions, 0)
    # charge = reduce(lambda sum, i: sum + i.charge, transactions)
    row.append(kWh)


def add_time_data(env: Environment, row):
    tod = env.current_tod
    if not tod:
        # in the first timeslot the current_tod is not yet set
        tod = env.first_tod
    if not tod:
        logging.warning("no tod {}".format(env.current_timestep))
        return
    row.append(tod.hour)
    row.append(tod.isoweekday())


def add_weather_data(env: Environment, row):
    # we are adding a random forecast from the forecasts we have + we tell the algorithm how far into the future this is
    # this way the algorithm learns to handle any kind of forecast distance and adapts the weight of it according to
    # the distance of the forecast
    # fc_dist = random.randint(1, 24)
    fc_dist = 24
    origin = env.current_timestep - fc_dist
    if (origin <= env.first_timestep):
        origin = env.first_timestep

    key = "{}+{}".format(origin, fc_dist)
    if key in env.weather_store.weather_predictions:
        fc = env.weather_store.weather_predictions[key]
        row.extend(fc.get_values_list()[:-1])  # ignoring origin value
    else:
        logging.warning("no forecast found! why?{} first: {}".format(key, env.first_timestep))

    # weather never has to be larger than 360.... But it's always 0 in the states anyways
    if row[15] > 360:
        raise ValueError


def add_rate_data(env: Environment, row, transactions: List[TariffTransaction]):
    first_consume = transactions[0]
    rate = env.tariff_store.get_rate_for_customer_transactions([first_consume])
    if not rate:
        return
    vals = rate.get_values_list()
    row.extend(vals[-5:])


def add_customer_data(customer: CustomerInfo, game):
    # getting customer metadata (first few columns)
    _ensure_customer(game, customer)
    row = customer.get_values_list()[2:8]  # adding only relevant things (ignoring customer powerType!
    del row[1]
    row[1] = row[1].value  # overwriting this as a raw value so we don't pickle objects
    return row


tick = 0


def tick_callback(env: Environment):
    global tick
    tick += 1
    if tick % 100 == 0:
        logging.info("tick {}".format(tick))
    # time_function(make_training_rows, [env])
    make_training_rows(env)
