from datetime import datetime, timedelta

from env.environment import Environment
from model.StatelineParser import StatelineParser
from util.config import DATETIME_PATTERN


class TimeslotStore:
    def __init__(self, env: Environment):
        self.env = env
        self.timeslots = {}


    def get_datetime_for_timeslot(self, timeslot: int) -> datetime:
        return self.env.first_tod + timedelta(hours=timeslot)

    def handle_timeslotUpdate_new(self, line: str):
        parts       = StatelineParser.split_line(line)
        date = parts[3]
        global first_enabled, last_enabled, current_timestep, current_tod

        current_tod = datetime.strptime(date, DATETIME_PATTERN)
        first_enabled = int(parts[4])
        last_enabled = int(parts[5])
        current_timestep = first_enabled - 1


