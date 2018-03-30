"""
A parser of state files to continuously parse one or many state files and feed
them into a consumer
"""
import os
import re
from typing import List

from model import environment
from util.function_timer import time_function

_root_dir = "/home/pascalwhoop/tank/Technology/Thesis/past_games"
logs_home = "extracted/"

ignored_states = set()


def run_through_all_files(tickcallback=None, roundcallback=None):
    """high level function to hook into parsing and get messages from environment"""
    files = get_state_files()
    for f in files:
        print("parsing file {}".format(f))
        states = get_states_from_file(f)

        parse_state_lines(states, tickcallback)

        if roundcallback is not None and callable(roundcallback):
            roundcallback()


def get_state_files(root_dir=_root_dir):
    """
    returns a list of state file paths
    """
    base = os.path.join(root_dir, logs_home)
    records = os.listdir(base)
    state_files = [os.path.join(base, r, "log", r + ".state") for r in records]
    return state_files


def get_states_from_file(file, states=None) -> List[str]:
    """Gets a list of specific class lines"""
    state_messages = []
    with open(file) as f:
        for line in f:
            if states is not None and any(s + "::" in line for s in states):
                state_messages.append(line)
            elif states is None:
                state_messages.append(line)

    return state_messages


def get_method(msg: str):
    return msg.split("::")[2]


def parse_state_lines(messages, callback=None):
    """
    Expects a list of messages (lines from the state logs) and updates the environment accordingly.
    It's important to note that the messages need to include all types needed to construct the
    tariffs. That also includes the ticks. It's best to just pass this the whole state file
    :param messages: state lines
    :return: list of
    """
    for msg in messages:
        # allow for callback after every timeslot is complete
        if "TimeslotUpdate" in msg and callback is not None and callable(callback):
            callback()

        msg = msg.strip()
        class_ = get_class(msg)
        method = get_method(msg)

        if class_ not in line_parsers:
            ignored_states.add((class_, method))
            continue

        class_handler = line_parsers[class_]
        if callable(class_handler):
            class_handler(msg)
        elif class_handler is not None and method in class_handler and callable(class_handler[method]):
            class_handler[method](msg)
        else:
            ignored_states.add((class_, method))


def get_class(msg: str):
    pattern = re.compile("(org[^:]*)")
    origin = pattern.search(msg)
    return origin.group(0)


"""Mapping between state origins and corresponding handlers. Most will be None but some are important to the agents
learning algorithms and will therefore be processed.

Received through `cat *.state | egrep "org[^:]*" -o | sort -u`
"""
line_parsers = {
    "org.powertac.common.BalancingTransaction": None,
    "org.powertac.common.BankTransaction": None,
    "org.powertac.common.Broker": None,
    "org.powertac.common.CapacityTransaction": None,
    "org.powertac.common.CashPosition": None,
    "org.powertac.common.ClearedTrade": None,
    "org.powertac.common.Competition": {
        "withBootstrapTimeslotCount": environment.handle_competition_withBootstrapTimeslotCount,
        "withBootstrapDiscardedTimeslots": environment.handle_competition_withBootstrapDiscardedTimeslots,
        "withSimulationBaseTime": environment.handle_competition_withSimulationBaseTime},
    "org.powertac.common.CustomerInfo": environment.handle_customerInfo,
    "org.powertac.common.DistributionTransaction": None,
    "org.powertac.common.MarketPosition": None,
    "org.powertac.common.MarketTransaction": None,
    "org.powertac.common.msg.BalanceReport": None,
    "org.powertac.common.msg.BalancingControlEvent": None,
    "org.powertac.common.msg.BalancingOrder": None,
    "org.powertac.common.msg.DistributionReport": None,
    "org.powertac.common.msg.EconomicControlEvent": None,
    "org.powertac.common.msg.OrderStatus": None,
    "org.powertac.common.msg.SimEnd": None,  # not needed
    "org.powertac.common.msg.SimPause": None,  # not needed
    "org.powertac.common.msg.SimResume": None,  # not needed
    "org.powertac.common.msg.SimStart": None,
    "org.powertac.common.msg.TariffRevoke": {"new": environment.handle_tariffRevoke_new},
    "org.powertac.common.msg.TariffStatus": {"new": environment.handle_tariffStatus_new},
    "org.powertac.common.msg.TimeslotUpdate": {"new": environment.handle_timeslotUpdate_new},
    "org.powertac.common.Order": None,
    "org.powertac.common.Orderbook": None,
    "org.powertac.common.OrderbookOrder": None,
    "org.powertac.common.RandomSeed": None,  # not needed
    "org.powertac.common.Rate": {"-rr": environment.handle_rate_rr, "new": environment.handle_rate_new,
                                 "withValue": environment.handle_rate_withValue,
                                 "setTariffId": environment.handle_rate_setTariffId},
    "org.powertac.common.RegulationCapacity": None,
    "org.powertac.common.RegulationRate": None,
    "org.powertac.common.Tariff": None,
    "org.powertac.common.TariffSpecification": {"new": environment.handle_tariff_new,
                                                "-rr": environment.handle_tariff_rr},
    "org.powertac.common.TariffSubscription": None,
    "org.powertac.common.TariffTransaction": {"new": environment.handle_TariffTransaction_new},
    "org.powertac.common.TimeService": None,
    "org.powertac.common.WeatherForecast": {"new": environment.handle_weatherForecast_new},  # not needed
    "org.powertac.common.WeatherForecastPrediction": {"new": environment.handle_weatherForecastPrediction_new},
    "org.powertac.common.WeatherReport": {"new": environment.handle_weatherReport_new},
    "org.powertac.du.DefaultBroker": None,
    "org.powertac.du.DefaultBrokerService": None,
    "org.powertac.evcustomer.customers.EvCustomer": None,
    "org.powertac.genco.Buyer": None,
    "org.powertac.genco.CpGenco": None,
    "org.powertac.genco.Genco": None,
    "org.powertac.genco.MisoBuyer": None,
}
