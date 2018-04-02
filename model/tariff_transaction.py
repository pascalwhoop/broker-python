"""
TariffTransaction model class. This class holds all sorts of things such as signups, costs caused, charges and benefits.

"""

from enum import Enum

from model.model_root import ModelRoot


class TransactionType(Enum):
    PUBLISH  = 1
    PRODUCE  = 2
    CONSUME  = 3
    PERIODIC = 4
    SIGNUP   = 5
    WITHDRAW = 6
    REVOKE   = 7
    REFUND   = 8


class TariffTransaction(ModelRoot):
    def __init__(self,
                 id_           = None,
                 when          = 0,
                 txType        = "CONSUME",
                 tariffSpec    = None,
                 customerInfo  = None,
                 customerCount = 0,
                 kWh           = 0.0,
                 charge        = 0.0,
                 regulation    = False
                ):

        self.id_           = id_
        self.when          = int(when)
        self.txType        = TransactionType[txType]
        self.customerInfo  = customerInfo
        self.customerCount = int(customerCount)
        self.kWh           = float(kWh)
        self.charge        = float(charge)
        self.regulation    = bool(regulation)
        self.tariffSpec    = tariffSpec



#  Type txType        = Type.CONSUME
#  CustomerInfo customerInfo
#  int customerCount  = 0
#  double kWh         = 0.0
#  double charge      = 0.0
#  boolean regulation = false
#  TariffSpecification tariffSpec
