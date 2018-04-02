"""
environment module for the agent. This holds anything that relates to state in the powertac environment
"""
from datetime import datetime, timedelta
from functools import reduce
from typing import List
import logging

from ast import literal_eval

import model.tariff as t
import model.tariff_status as ts
import model.customer_info as ci
import model.tariff_transaction as tt
from config import DATETIME_PATTERN
from model.tariff_transaction import TransactionType
from model.rate import Rate
from model.StatelineParser import StatelineParser
from model.weather import WeatherForecastPrediction, WeatherForecast, WeatherReport

current_timestep = 0
first_timestep   = 0
current_tod      = None
first_tod        = None
first_enabled    = 0
last_enabled     = 0

rates               = {}
tariffs             = {}
brokers             = {}
customers           = {}
weather_forecasts   = {}
weather_predictions = {} #keys are origin+FC --> "360+14" --> Obj
weather_reports     = {}
timeslots           = {}
transactions        = {}         # map of lists . Key => customerId, values the transaction objects

# holds stats about tariffs. each tariff holds the following information:
# alive_since: number of first timeslot offered
# produced_list: an array of production values of timeslots
# consumed_list: an array of consume values of timeslots
tariff_stats = {}


def get_rate_for_customer_transactions(transactions: List[tt.TariffTransaction]) -> Rate:

    potential_rates = [tariffs[t.tariffSpec]._rates for t in transactions]  #tariff and TariffSpec have same id (luckily)
    potential_rates = reduce(lambda sum, list: sum+list, potential_rates)
    time_of_transaction = get_datetime_for_timeslot(transactions[0].when)
    potential_rates = [r for r in potential_rates if r.is_applicable(time_of_transaction)]

    #no rates found... odd. manually searching and adding again
    if len(potential_rates) is 0:
        for t in transactions:
            potential_rates = [r for r in rates.values() if r.tariffId == t.tariffSpec]
            map(lambda r: tariffs[rates[0].tariffId].add_rate(r), potential_rates)

    if len(potential_rates) is 0:
        logging.warning("we're missing rates over here")
        return None
    # tier threshold allows multiple rates. gotta think how to handle this
    #if len(potential_rates)> 1:
    #    print("too many rates?")

    #    for r in potential_rates:
    #        print(r)
    #    print("for transactions")
    #    for t in transactions:
    #        print(t)
    return potential_rates[0]


# handling tariff messages
#########################################################################################################################

def get_datetime_for_timeslot(timeslot: int) -> datetime:
    return first_tod + timedelta(hours=timeslot)


_rates_left_over = []   #sometimes a rate is applied to multiple Tariffs in the state files. That causes confusion
                        #furthermore, the rate.-rr is called before the tariffSpec.-rr so I can't add the rate to the tariff
                        #if the tariff doesn't exist yet
def handle_tariff_rr(line: str):
    tariff = t.Tariff.from_state_line(line)
    tariffs[tariff.id_] = tariff
    for r in _rates_left_over:
        if r.tariffId == tariff.id_:
            tariff.add_rate(r)


def handle_tariff_new(line: str):
    parts = StatelineParser.split_line(line)
    tariff = t.Tariff(id_ = parts[1], brokerId =parts[3], powerType=parts[4])
    tariffs[tariff.id_] = tariff
    _add_tariff_stats_for_tariff_id(tariff.id_)


def handle_tariff_addRate(line: str):
    parts = StatelineParser.split_line(line)
    tariff = tariffs[parts[1]]
    tariff.add_rate(rates[3])


def handle_tariffStatus_new(line: str):
    _ts = ts.TariffStatus.from_state_line(line)
    if _ts.status == ts.Status.success:
        _add_tariff_stats_for_tariff_id(_ts.tariff_id)
        tariffs[_ts.tariff_id].status = t.Status.ACTIVE


def _add_tariff_stats_for_tariff_id(id_):
    tariff_stats[id_] = t.TariffStats(current_timestep)


def handle_tariffRevoke_new(line: str):
    parts = StatelineParser.split_line(line)
    tariffs[parts[4]].status = t.Status.WITHDRAWN

# handling rate messages
#########################################################################################################################

def handle_rate_new(line: str):
    parts = StatelineParser.split_line(line)
    id_ = parts[1]
    rates[id_] = Rate(id_)


def handle_rate_withValue(line: str):
    parts = StatelineParser.split_line(line)
    rate = rates[parts[1]]
    if rate is not None:
        rate.minValue = float(parts[3])


def handle_rate_setTariffId(line: str):
    parts = StatelineParser.split_line(line)
    rate = rates[parts[1]]
    if rate is not None:
        rate.tariffId = parts[3]
        tariffs[rate.tariffId].add_rate(rate)


def handle_rate_rr(line: str):
    rate = Rate.from_state_line(line)
    rates[rate.id_] = rate
    if rate.tariffId in tariffs:
        tariffs[rate.tariffId].add_rate(rate)
    else:
        _rates_left_over.append(rate)

# handling CustomerInfo
#########################################################################################################################

def handle_customerInfo(line: str):
    """
    handles all customerInfo methods. If it's a "new" method, we create the object
    and add it to the repo of customers. Otherwise the line gives the method and the params
    that are needed and we just call that. all methods are prefixed with _jsm_ which stands for
    "java state method". this is internal only and meant for learning off the state messages.

    """
    parts       = StatelineParser.split_line(line)
    method      = parts[2]
    method_call = "_jsm_" + parts[2]
    if method == "new":
        info = ci.CustomerInfo(id_ = parts[1], name=parts[3], population=parts[4])
        customers[parts[1]] = info
    else:
        target = customers[parts[1]]
        to_call = getattr(target, method_call)
        if callable(to_call):
            to_call(*parts[3:])


# handling TariffTransaction
#########################################################################################################################
def handle_TariffTransaction_new(line:str):

    # "8828:org.powertac.common.TariffTransaction::3328::new::1876          ::360       ::SIGNUP        ::1878                      ::3320                  ::30000             ::0.0           ::-0.0          ::false"
    # 100100:org.powertac.common.TariffTransaction::5486::new::1876         ::360       ::CONSUME       ::1878                      ::3709                  ::1                 ::-46.5874647092::23.2937423546::false
    # 100300:org.powertac.common.TariffTransaction::5492::new::1876         ::360       ::CONSUME       ::1878                      ::3728                  ::1                 ::-53.3333333333::26.66666666664::false
    # 100331:org.powertac.common.TariffTransaction::5498::new::1876         ::360       ::CONSUME       ::1878                      ::3742                  ::1                 ::-35.555555556 ::17.77777777778::false
    #                                                         Broker broker :: int when ::Type txType   ::TariffSpecification spec  :: CustomerInfo customer:: int customerCount:: double kWh   :: double charge:: boolean regulation)

    parts       = StatelineParser.split_line(line)
    type_ = _get_tariff_type(parts)
    params = [parts[1]] #id of object
    params.extend(parts[4:]) #params of original constructor
    transaction = tt.TariffTransaction(*params)
    _add_transaction(transaction)

    if   type_ is TransactionType.PUBLISH:
        # handle type. Technically here we need to set tariff to published
        pass
    elif type_ is TransactionType.PRODUCE or TransactionType.CONSUME or TransactionType.SIGNUP or TransactionType.WITHDRAW or TransactionType.REFUND :
        _handle_transaction_stats(transaction)
    elif type_ is TransactionType.PERIODIC:
        # handle type
        pass
    elif type_ is TransactionType.SIGNUP  :
        # handle type
        pass
    elif type_ is TransactionType.WITHDRAW:
        # handle type
        pass
    elif type_ is TransactionType.REVOKE  :
        # handle type
        pass
    elif type_ is TransactionType.REFUND  :
        # handle type
        pass


def _add_transaction(trans: tt.TariffTransaction):
    """gets always called (we store everything)"""
    customer_id = trans.customerInfo
    # initializing on first transaction for customer
    if customer_id not in transactions:
        transactions[customer_id] = []
    cust_trans = transactions[customer_id]
    # adding new list for current timestep
    if len(cust_trans)<current_timestep or not cust_trans:
        cust_trans.append([])

    trans_now = cust_trans[-1]
    trans_now.append(trans)


def _handle_transaction_stats(trans: tt.TariffTransaction):
    """calculates specific statistics about the tariff in question"""
    tariff_stat = tariff_stats[trans.tariffSpec]
    tariff_stat.apply_tariff_transaction(trans)

    


def _get_tariff_type(parts: List[str]) -> TransactionType:
    return TransactionType[parts[5]]


# operations on environment
#########################################################################################################################


def reset():
    global current_timestep , first_timestep, current_tod , first_tod , first_enabled , last_enabled , rates , tariffs , brokers , customers , weather_forecasts , weather_predictions , weather_reports , timeslots , transactions 
    current_timestep = 0
    first_timestep   = 0
    current_tod      = None
    first_tod        = None
    first_enabled    = 0
    last_enabled     = 0
    
    rates               = {}
    tariffs             = {}
    brokers             = {}
    customers           = {}
    weather_forecasts   = {}
    weather_predictions = {} #keys are origin+FC --> "360+14" --> Obj
    weather_reports     = {}
    timeslots           = {}
    transactions        = {}         # map of lists. Key => customerId, values the transaction objects

#competition
#########################################################################################################################
def handle_competition_withBootstrapTimeslotCount(line: str):
    parts       = StatelineParser.split_line(line)
    global current_timestep, first_timestep
    current_timestep = int(parts[3])
    first_timestep = current_timestep

def handle_competition_withBootstrapDiscardedTimeslots(line:str):
    parts       = StatelineParser.split_line(line)
    global current_timestep, first_timestep
    current_timestep += int(parts[3])
    first_timestep    = current_timestep





def handle_competition_withSimulationBaseTime(line: str):
    parts       = StatelineParser.split_line(line)
    timestamp = parts[3]
    start = datetime.fromtimestamp(int(int(timestamp)/1000)) #java returns milli fromtimestamp takes seconds
    global first_tod
    first_tod = start


def handle_timeslotUpdate_new(line: str):
    parts       = StatelineParser.split_line(line)
    date = parts[3]
    global first_enabled, last_enabled, current_timestep, current_tod

    current_tod = datetime.strptime(date, DATETIME_PATTERN)
    first_enabled = int(parts[4])
    last_enabled = int(parts[5])
    current_timestep = first_enabled - 1
# weather handlers
#########################################################################################################################
def handle_weatherForecastPrediction_new(line: str):
    parts       = StatelineParser.split_line(line)
    key = parts[1]
    prediction = WeatherForecastPrediction.from_state_line(line)
    prediction.origin = current_timestep
    weather_predictions[key] = prediction

def handle_weatherForecast_new(line:str):
    #6651:org.powertac.common.WeatherForecast::602::new::360::(26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49)
    parts       = StatelineParser.split_line(line)
    origin = parts[-2]
    ids    = parts[-1]
    ids = literal_eval(ids)
    for id in ids:
        key = str(id)
        if key in weather_predictions:
            prediction = weather_predictions.pop(key)
            weather_predictions["{}+{}".format(origin,prediction.forecastTime)] = prediction
        else:
            logging.warning("weather not found {} -- {} -- {}".format(key, origin, id))


def handle_weatherReport_new(line: str):
    report = WeatherReport.from_state_line(line)
    weather_reports[report.currentTimeslot] = report


    #TODO keep reports
    return None



