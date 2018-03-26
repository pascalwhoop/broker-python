"""
Holds the customer data that is passed to the brokers during bootstrap of the game.
"""
from enum import Enum

from model.model_root import ModelRoot


class CustomerClass(Enum):
    SMALL = 1
    LARGE = 2


class CustomerInfo(ModelRoot):

    def __init__(self,
                 id_              = None,
                 name             = None,
                 population       = 0,
                 powerType        = {0}, # default CONSUMPTION
                 customerClass    = CustomerClass.SMALL,
                 controllableKW   = 0.0,
                 upRegulationKW   = 0.0,
                 downRegulationKW = 0.0,
                 storageCapacity  = 0.0,
                 multiContracting = False,
                 canNegotiate     = False):

        self.id_              = id_
        self.name             = name
        self.population       = population
        self.powerType        = powerType
        self.customerClass    = customerClass
        self.controllableKW   = controllableKW
        self.upRegulationKW   = upRegulationKW
        self.downRegulationKW = downRegulationKW
        self.storageCapacity  = storageCapacity
        self.multiContracting = multiContracting
        self.canNegotiate     = canNegotiate

    def _jsm_setPopulation(self,population: int):
        self.population = int(population)

    def gjsm_withPowerType(self,type_):
        self.powerType.add(type_)

    def _jsm_withCustomerClass(self,cClass: CustomerClass):
        self.customerClass = cClass

    def _jsm_withMultiContracting(self,value: bool):
        self.multiContracting = bool(value)

    def _jsm_withCanNegotiate(self,value: bool):
        self.canNegotiate = bool(value)

    def _jsm_withControllableKW(self,value: float):
        self.controllableKW = float(value)

    def _jsm_withUpRegulationKW(self,value: float):
        self.upRegulationKW = float(value)

    def _jsm_withDownRegulationKW(self,value: float):
        self.downRegulationKW = float(value)

    def _jsm_withStorageCapacity(self,value: float):
        self.storageCapacity = float(value)


