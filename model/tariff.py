"""
Tariff type, holds base information about a tariff offered by a broker
TODO will this also hold the rate that is linked to it?
"""
from enum    import Enum
from typing  import List
import numpy as np
from util import config as cfg
from model.rate import Rate
from model.tariff_transaction import TariffTransaction, TransactionType
from model.StatelineParser import StatelineParser

LENGTH_STATS = 8

class TariffStats():
    def __init__(self, initial_timeslot = 0):
        """Tariff statsu constructor
        :returns: TODO

        """
        self.initial_timeslot = initial_timeslot

        """
        timeslot_stats are structured as follows:

        [0]positive_charge_sum = 0.0 <-- not yet
        [1]negative_charge_sum = 0.0 <-- not yet
        [2]charge_sum          = 0.0
        [3]positive_kWh_sum    = 0.0 <-- not yet
        [4]negative_kWh_sum    = 0.0 <-- not yet
        [5]kWh_sum             = 0.0
        [6]onetime_sum         = 0.0
        [7]periodic_sum        = 0.0


        """
        self.timeslot_stats   = []

    def get_timeslot_stats(self, ts: int) -> List[float]:
        return self.timeslot_stats[ts-self.initial_timeslot]

    def apply_tariff_transaction(self, tt: TariffTransaction):
        """handles all those tariff transactions that effect the kWh and charge as well as signup and signoff"""
        index = tt.when - self.initial_timeslot

        # if the timeslot has not been calculated yet, add a new one
        while len(self.timeslot_stats) < index + 1:
            self.timeslot_stats.append(np.zeros(LENGTH_STATS))

        ts_stats = self.timeslot_stats[index]
        if tt.txType is TransactionType.CONSUME or TransactionType.PRODUCE:
            ts_stats[2] += round(tt.charge, 4)
            ts_stats[5] += round(tt.kWh, 4)
        elif tt.txType is TransactionType.PERIODIC:
            ts_stats[7] += round(tt.charge, 4)
        elif tt.txType is TransactionType.SIGNUP or TransactionType.WITHDRAW:
            ts_stats[6] += round(tt.charge, 4)





class Status(Enum):
    """
    variants of tariff status.
    """
    PENDING = 1
    OFFERED = 2
    ACTIVE = 3
    WITHDRAWN = 4
    KILLED = 5


class Tariff(StatelineParser):
    """
    Related to this https://github.com/powertac/powertac-server/wiki/Tariff-representation
    """

    def __init__(self                 = None,
                 id_                  = "",
                 brokerId             = "",
                 powerType            = "",
                 minDuration          = 0,
                 signupPayment        = 0.0,
                 earlyWithdrawPayment = 0.0,
                 periodicPayment      = 0.0):
        """
            From the JAVA_DOCS
             * State log fields for readResolve():<br>
             * <code>long brokerId, PowerType powerType, long minDuration,<br>
             * &nbsp;&nbsp;double signupPayment, double earlyWithdrawPayment,<br>
             * &nbsp;&nbsp;double periodicPayment, List<tariffId> supersedes</code>

        """
        self.id_                  = id_
        self.status               = Status.PENDING
        self.brokerId             = brokerId
        self.powerType            = powerType
        self.minDuration          = minDuration
        self.signupPayment        = signupPayment
        self.earlyWithdrawPayment = earlyWithdrawPayment
        self.periodicPayment      = periodicPayment

        # python specific
        self._rates = []

    def add_rate(self, rate: Rate):
        self._rates.append(rate)

    @staticmethod
    def from_state_line(line: str) -> "Tariff":
        parts = StatelineParser.split_line(line)
        return Tariff(parts[1],
                      parts[3],
                      parts[4],
                      int(parts[5]),
                      round(float(parts[6]),
                            cfg.ROUNDING_PRECISION),
                      round(float(parts[7]),
                            cfg.ROUNDING_PRECISION),
                      round(float(parts[8]),
                            cfg.ROUNDING_PRECISION))

    def is_active(self) -> bool:
        return self.status is Status.ACTIVE
