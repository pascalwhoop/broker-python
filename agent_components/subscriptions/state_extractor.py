"""
A parser of state files to continuously parse one or many state files and feed
them into a consumer
"""
import os
import re

from model import environment

_root_dir = "/home/pascalwhoop/tank/Technology/Thesis/past_games"
logs_home = "extracted/"

ignored_states = set()


def get_state_files(root_dir=_root_dir):
    """
    returns a list of state file paths
    """
    base = os.path.join(root_dir, logs_home)
    records = os.listdir(base)
    state_files = [os.path.join(base, r, "log", r + ".state") for r in records]
    return state_files


def get_states_from_file(file, states=None):
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


def parse_state_lines(messages):
    """
    Expects a list of messages (lines from the state logs) and generates a repository of tariffs that have existed
    throughout the game. It's important to note that the messages need to include all types needed to construct the
    tariffs. That also includes the ticks
    :param messages: state lines
    :return: list of
    """
    timestep = 0
    for msg in messages:
        msg = msg.strip()
        class_ = get_class(msg)
        method = get_method(msg)
        try:  # why tryexcept?
            line_parsers[class_][method](msg)
        except:
            ignored_states.add((class_, method))
            pass


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
    "org.powertac.common.Competition": None,
    "org.powertac.common.CustomerInfo": None,
    "org.powertac.common.DistributionTransaction": None,
    "org.powertac.common.MarketPosition": None,
    "org.powertac.common.MarketTransaction": None,
    "org.powertac.common.msg.BalanceReport": None,
    "org.powertac.common.msg.BalancingControlEvent": None,
    "org.powertac.common.msg.BalancingOrder": None,
    "org.powertac.common.msg.DistributionReport": None,
    "org.powertac.common.msg.EconomicControlEvent": None,
    "org.powertac.common.msg.OrderStatus": None,
    "org.powertac.common.msg.SimEnd": None,
    "org.powertac.common.msg.SimPause": None,
    "org.powertac.common.msg.SimResume": None,
    "org.powertac.common.msg.SimStart": None,
    "org.powertac.common.msg.TariffRevoke": {"new": environment.handle_tariff_revoke_from_state_line},
    "org.powertac.common.msg.TariffStatus": None,
    "org.powertac.common.msg.TimeslotUpdate": None,
    "org.powertac.common.Order": None,
    "org.powertac.common.Orderbook": None,
    "org.powertac.common.OrderbookOrder": None,
    "org.powertac.common.RandomSeed": None,
    "org.powertac.common.Rate": None,
    "org.powertac.common.RegulationCapacity": None,
    "org.powertac.common.RegulationRate": None,
    "org.powertac.common.Tariff": None,
    "org.powertac.common.TariffSpecification": {"-rr": environment.add_tariff_from_state_line},
    "org.powertac.common.TariffSubscription": None,
    "org.powertac.common.TariffTransaction": None,
    "org.powertac.common.TimeService": None,
    "org.powertac.common.WeatherForecast": None,
    "org.powertac.common.WeatherForecastPrediction": None,
    "org.powertac.common.WeatherReport": None,
    "org.powertac.customer.coldstorage.ColdStorage": None,
    "org.powertac.customer.coldstorage.ColdStorage-freezeco-1": None,
    "org.powertac.customer.coldstorage.ColdStorage-freezeco-2": None,
    "org.powertac.customer.coldstorage.ColdStorage-freezeco-3": None,
    "org.powertac.customer.coldstorage.ColdStorage-seafood-1": None,
    "org.powertac.customer.coldstorage.ColdStorage-seafood-2": None,
    "org.powertac.customer.model.Battery": None,
    "org.powertac.customer.model.Battery-b1": None,
    "org.powertac.customer.model.Battery-b10": None,
    "org.powertac.customer.model.Battery-b11": None,
    "org.powertac.customer.model.Battery-b12": None,
    "org.powertac.customer.model.Battery-b13": None,
    "org.powertac.customer.model.Battery-b14": None,
    "org.powertac.customer.model.Battery-b15": None,
    "org.powertac.customer.model.Battery-b16": None,
    "org.powertac.customer.model.Battery-b17": None,
    "org.powertac.customer.model.Battery-b18": None,
    "org.powertac.customer.model.Battery-b19": None,
    "org.powertac.customer.model.Battery-b2": None,
    "org.powertac.customer.model.Battery-b20": None,
    "org.powertac.customer.model.Battery-b21": None,
    "org.powertac.customer.model.Battery-b22": None,
    "org.powertac.customer.model.Battery-b23": None,
    "org.powertac.customer.model.Battery-b24": None,
    "org.powertac.customer.model.Battery-b25": None,
    "org.powertac.customer.model.Battery-b26": None,
    "org.powertac.customer.model.Battery-b27": None,
    "org.powertac.customer.model.Battery-b28": None,
    "org.powertac.customer.model.Battery-b29": None,
    "org.powertac.customer.model.Battery-b3": None,
    "org.powertac.customer.model.Battery-b30": None,
    "org.powertac.customer.model.Battery-b4": None,
    "org.powertac.customer.model.Battery-b5": None,
    "org.powertac.customer.model.Battery-b6": None,
    "org.powertac.customer.model.Battery-b7": None,
    "org.powertac.customer.model.Battery-b8": None,
    "org.powertac.customer.model.Battery-b9": None,
    "org.powertac.customer.model.LiftTruck": None,
    "org.powertac.customer.model.LiftTruck-fc2": None,
    "org.powertac.customer.model.LiftTruck-fc3": None,
    "org.powertac.customer.model.LiftTruck-sf2": None,
    "org.powertac.customer.model.LiftTruck-sf3": None,
    "org.powertac.du.DefaultBroker": None,
    "org.powertac.du.DefaultBrokerService": None,
    "org.powertac.evcustomer.customers.EvCustomer": None,
    "org.powertac.genco.Buyer": None,
    "org.powertac.genco.CpGenco": None,
    "org.powertac.genco.Genco": None,
    "org.powertac.genco.MisoBuyer": None,
}
