from enum import Enum

from model.StatelineParser import StatelineParser


class Status(Enum):
    success          = 1
    noSuchTariff     = 2
    noSuchUpdate     = 3
    illegalOperation = 4
    invalidTariff    = 5
    invalidUpdate    = 6
    duplicateId      = 7
    invalidPowerType = 8
    unsupported      = 9


class TariffStatus(StatelineParser):
    def __init__(self, status, tariff_id):
        self.status = status
        self.tariff_id = tariff_id

    @staticmethod
    def from_state_line(line: str) -> "TariffStatus":
        parts = StatelineParser.split_line(line)
        return TariffStatus(Status[parts[6]], parts[4])
