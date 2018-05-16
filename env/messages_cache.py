"""
These are all the stores for various messages that we receive from the server. Caching all of them until we clear them. 
"""
import sys

from pydispatch import dispatcher

PBActivateCache                = []
PBBalanceReportCache           = []
PBBalancingControlEventCache   = []
PBBalancingTransactionCache    = []
PBBankTransactionCache         = []
PBCapacityTransactionCache     = []
PBCashPositionCache            = []
PBClearedTradeCache            = []
PBCompetitionCache             = []
PBCustomerBootstrapDataCache   = []
PBDistributionReportCache      = []
PBDistributionTransactionCache = []
PBMarketBootstrapDataCache     = []
PBMarketPositionCache          = []
PBMarketTransactionCache       = []
PBOrderCache                   = []
PBOrderbookCache               = []
PBPropertiesCache              = []
PBSimPauseCache                = []
PBSimResumeCache               = []
PBTariffRevokeCache            = []
PBTariffSpecificationCache     = []
PBTariffStatusCache            = []
PBTariffTransactionCache       = []
PBTimeslotCompleteCache        = []
PBTimeslotUpdateCache          = []
PBWeatherForecastCache         = []
PBWeatherReportCache           = []

def store_message(sender, signal, msg):
    cache_name = signal+"Cache"
    if cache_name in dir(sys.modules[__name__]):
        globals()[cache_name].append(msg)


dispatcher.connect(store_message, signal=dispatcher.Any, sender=dispatcher.Any)
