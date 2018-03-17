"""
Tariff type, holds base information about a tariff offered by a broker
TODO will this also hold the rate that is linked to it?
"""
from enum import Enum

import Config as cfg
from model.StatelineParser import StatelineParser


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

    def __init__(self, _id, broker_id, power_type, min_duration, signup_payment,
                 early_withdraw_payment, periodic_payment):
        """
            From the JAVA_DOCS
             * State log fields for readResolve():<br>
             * <code>long brokerId, PowerType powerType, long minDuration,<br>
             * &nbsp;&nbsp;double signupPayment, double earlyWithdrawPayment,<br>
             * &nbsp;&nbsp;double periodicPayment, List<tariffId> supersedes</code>

        """
        self.id = _id
        self.status = Status.PENDING
        self.broker_id = broker_id
        self.power_type = power_type
        self.min_duration = min_duration
        self.signup_payment = signup_payment
        self.early_withdraw_payment = early_withdraw_payment
        self.periodic_payment = periodic_payment

    @staticmethod
    def from_state_line(line: str) -> "Tariff":
        parts = StatelineParser.split_line(line)
        return Tariff(parts[1], parts[3], parts[4], int(parts[5]), round(float(parts[6]), cfg.ROUNDING_PRECISION),
                      round(float(parts[7]), cfg.ROUNDING_PRECISION), round(float(parts[8]), cfg.ROUNDING_PRECISION))
