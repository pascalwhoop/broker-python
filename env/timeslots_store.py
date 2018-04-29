from datetime import datetime, timedelta

from model.StatelineParser import StatelineParser
from util.config import DATETIME_PATTERN


class TimeslotStore:
    def __init__(self, env):
        self.env = env
        self.timeslots = {}


    def get_datetime_for_timeslot(self, timeslot: int) -> datetime:
        return self.env.first_tod + timedelta(hours=timeslot)

    def handle_timeslotUpdate_new(self, line: str):
        parts       = StatelineParser.split_line(line)
        date = parts[3]

        self.env.current_tod      = datetime.strptime(date, DATETIME_PATTERN)
        self.env.first_enabled    = int(parts[4])
        self.env.last_enabled     = int(parts[5])
        self.env.current_timestep = self.env.first_enabled - 1
