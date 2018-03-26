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
import pickle
from functools import reduce
from typing import List

import model.environment as env
from model.customer_info import CustomerInfo
from model.tariff_transaction import TransactionType, TariffTransaction
import util.state_extractor as se

# holds time series for consume data for several customers
# game []
#    customer {}
#       [training, result]
from util.function_timer import time_function

consume_data = [{}]


game_counter = 0
def round_callback():
    env.reset()
    consume_data.append({})
    global game_counter
    game_counter +=1




def _ensure_customer(game, customer: CustomerInfo):
    if customer.id_ not in game:
        #list with 2 lists. Samples and Results (what the customer used
        game[customer.id_] = [[], []]

def make_training_rows(env):
    """iterating over the current snapshot of the state of the game and getting relevant data to build samples"""
    game = consume_data[game_counter]

    #we take a sample for each customer at each timeslot
    for customer in env.customers.values():
        row = add_customer_data(customer, game)

        """if we want to learn also based on type of tariff --> add this"""
        ##getting customer tariff data
        #for t in reversed(transactions[customer.id_]):
        #    if t.txType == TransactionType.SIGNUP:
        #        tariff = env.tariffs[t.tariffSpec]

        #getting rate applicable (next few rows)
        customer_transactions = env.transactions[customer.id_]
        if len(customer_transactions) == 0:
            return
        transactions = [t for t in customer_transactions[-1] if t.txType == TransactionType.CONSUME or TransactionType.PRODUCE]
        add_rate_data(row, transactions)

        #getting consume of today
        add_consume_data(row, transactions)

        #let's check if this is worth adding. if so, we add the input to list0 and the result to list1
        #[0,6] customer_data
        #[7,
        if len(row) is 12:
            game[customer.id_][0].append(row[0:-1])
            game[customer.id_][1].append(row[-1])


def add_consume_data(row, transactions:List[TariffTransaction]):
    kWh = reduce(lambda sum, i: sum + i.kWh, transactions, 0)
    # charge = reduce(lambda sum, i: sum + i.charge, transactions)
    row.append(kWh)


def add_rate_data(row, transactions: List[TariffTransaction]):
    txs = [t for t in transactions if t.txType == TransactionType.CONSUME]
    # if no transaction of type consume existed yet, no rate --> no extension
    if not txs:
        return
    first_consume = txs[0]
    rate = env.get_rate_for_customer_transaction(first_consume)
    if not rate:
        return
    vals = rate.get_values_list()
    row.extend(vals[-5:])


def add_customer_data(customer: CustomerInfo, game):
    # getting customer metadata (first few columns)
    _ensure_customer(game, customer)
    row = customer.get_values_list()[2:8] #adding only relevant things
    return row


tick = 0
def tick_callback():
    global tick
    tick += 1
    print("tick {}".format(tick))
    time_function(make_training_rows, [env])
    #make_training_rows(env)
