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
from env.tariff_store import TariffStore
from env.timeslots_store import TimeslotStore
from env.weather_store import WeatherStore
from util.config import DATETIME_PATTERN
from model.tariff_transaction import TransactionType
from model.rate import Rate
from model.StatelineParser import StatelineParser

_env = None
def get_instance():
    """manage environment as singleton"""
    global _env
    if _env is None:
        _env = Environment()
    return _env

def reset_instance():
    #TODO is this applying itself to all those that hold a reference to the instance pointer?
    global _env
    del _env
    _env = Environment()

class Environment():
    def __init__(self, ):
        self.current_timestep = 0
        self.first_timestep   = 0
        self.current_tod      = None
        self.first_tod        = None
        self.first_enabled    = 0
        self.last_enabled     = 0
        #repos
        self.weather_store = WeatherStore(self)
        self.tariff_store = TariffStore(self)
        self.timeslot_store = TimeslotStore(self)










    #competition
    #########################################################################################################################
    def handle_competition_withBootstrapTimeslotCount(self, line: str):
        parts       = StatelineParser.split_line(line)
        global current_timestep, first_timestep
        current_timestep = int(parts[3])
        first_timestep = current_timestep

    def handle_competition_withBootstrapDiscardedTimeslots(self, line:str):
        parts       = StatelineParser.split_line(line)
        global current_timestep, first_timestep
        current_timestep += int(parts[3])
        first_timestep    = current_timestep





    def handle_competition_withSimulationBaseTime(self, line: str):
        parts       = StatelineParser.split_line(line)
        timestamp = parts[3]
        start = datetime.fromtimestamp(int(int(timestamp)/1000)) #java returns milli fromtimestamp takes seconds
        global first_tod
        first_tod = start



