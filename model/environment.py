"""
environment module for the agent. This holds anything that relates to state in the powertac environment
"""
import model.tariff as t
import model.tariff_status as ts
from model.StatelineParser import StatelineParser

tariffs = {}
brokers = {}
customers = {}
weather_forecasts = {}
weather_reports = {}
timeslots = {}


def add_tariff_from_state_line(line: str):
    tariff = t.Tariff.from_state_line(line)
    tariffs[tariff.id] = tariff


def handle_tariff_status_from_state_line(line: str):
    _ts = ts.TariffStatus.from_state_line(line)
    if _ts.status == ts.Status.success:
        tariffs[_ts.tariff_id].status = t.Status.ACTIVE


def handle_tariff_revoke_from_state_line(line: str):
    parts = StatelineParser.split_line(line)
    tariffs[parts[4]].status = t.Status.WITHDRAWN


def handle_rate_from_state_line(line: str):
    #TODO
    pass
