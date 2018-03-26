"""
environment module for the agent. This holds anything that relates to state in the powertac environment
"""
from datetime import datetime, timedelta
from typing import List


import model.tariff as t
import model.tariff_status as ts
import model.customer_info as ci
import model.tariff_transaction as tt
from config import DATETIME_PATTERN
from model.tariff_transaction import TransactionType
from model.rate import Rate
from model.StatelineParser import StatelineParser
from model.weather import WeatherForecastPrediction, WeatherForecast, WeatherReport

current_timestep  = 0
current_tod       = None
first_tod         = None
first_enabled     = 0
last_enabled      = 0

rates             = {}
tariffs           = {}
brokers           = {}
customers         = {}
weather_forecasts = {}
weather_predictions       = {} #keys are origin+FC --> "360+14" --> Obj
weather_reports   = {}
timeslots         = {}
transactions      = {}         # map of lists. Key => customerId, values the transaction objects

# holds stats about tariffs. each tariff holds the following information:
# alive_since: number of first timeslot offered
# produced_list: an array of production values of timeslots
# consumed_list: an array of consume values of timeslots
tariff_stats = {}


def get_rate_for_customer_transaction(transaction: tt.TariffTransaction) -> Rate:
    potential_rates = [r for r in rates.values() if r.tariffId == transaction.tariffSpec] #tariff and TariffSpec have same id (luckily)
    time_of_transaction = get_datetime_for_timeslot(transaction.when)
    potential_rates = [r for r in potential_rates if r.is_applicable(time_of_transaction)]
    if len(potential_rates) > 1:
        print("something is weird with the timeslots here")
    elif len(potential_rates) is 0:
        return None
    return potential_rates[0]


# handling tariff messages
#########################

def get_datetime_for_timeslot(timeslot: int) -> datetime:
    return first_tod + timedelta(hours=timeslot)


def handle_tariff_rr(line: str):
    tariff = t.Tariff.from_state_line(line)
    tariffs[tariff.id] = tariff


def handle_tariffStatus_new(line: str):
    _ts = ts.TariffStatus.from_state_line(line)
    if _ts.status == ts.Status.success:
        tariff_stats[_ts.tariff.id] = t.TariffStats(current_timestep)
        tariffs[_ts.tariff_id].status = t.Status.ACTIVE


def handle_tariffRevoke_new(line: str):
    parts = StatelineParser.split_line(line)
    tariffs[parts[4]].status = t.Status.WITHDRAWN

# handling rate messages
#########################

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


# handling CustomerInfo
#########################

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
        print(info)
        customers[parts[1]] = info
    else:
        target = customers[parts[1]]
        to_call = getattr(target, method_call)
        if callable(to_call):
            to_call(*parts[3:])


# handling TariffTransaction
############################
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

# all the music is here
    # 1. record the amount of consume per timeslot / tariff
    # 2. record the accumulated consume per timeslot / tariff
    # 3. record all prices of produce / consume / regulate / ... for all tariffs
    # keep running average of price
    # ...

def _add_transaction(trans: tt.TariffTransaction):
    """gets always called (we store everything)"""
    customer_id = trans.customerInfo
    if customer_id not in transactions:
        transactions[customer_id] = []
    transactions[customer_id].append(trans)


def _handle_transaction_stats(trans: tt.TariffTransaction):
    """calculates specific statistics about the tariff in question"""
    tariff      = tariffs[trans.tariffSpec]
    tariff_stat = tariff_stats[trans.tariffSpec]
    tariff_stat.apply_tariff_transaction(trans)

    


def _get_tariff_type(parts: List[str]) -> TransactionType:
    return TransactionType[parts[5]]

def reset():
    global rates, tariffs, brokers, customers, weather_forecasts, weather_reports, timeslots, transactions
    rates             = {}
    tariffs           = {}
    brokers           = {}
    customers         = {}
    weather_forecasts = {}
    weather_reports   = {}
    timeslots         = {}
    transactions      = {}


def handle_weatherForecastPrediction_new(line: str):
    prediction = WeatherForecastPrediction.from_state_line(line)
    prediction.origin = current_timestep
    #key: 360+15 --> origin is 360, FCtime +15
    key = "{}+{}".format(prediction.origin, prediction.forecastTime)
    weather_predictions[key] = prediction

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
    current_timestep += 1


def handle_weatherReport_new(line: str):
    parts       = StatelineParser.split_line(line)
    report = WeatherReport.from_state_line(line)
    weather_reports[report.currentTimeslot] = report


    #TODO keep reports
    return None

