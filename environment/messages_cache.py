"""
These are all the stores for various messages that we receive from the server. Caching all of them until we clear them. 
"""
import pickle

import json
import os
import sys

from google.protobuf.json_format import MessageToJson
from pydispatch import dispatcher
import util.config as cfg
from util.utils import get_now_date_file_ready

_path = os.path.join(cfg.DATA_LOG_PATH, get_now_date_file_ready())
file_handlers = {}

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
        #also logging it to file
    log_message(signal, msg)

# ------
# hook into pubsub

def subscribe():
    dispatcher.connect(store_message, signal=dispatcher.Any, sender=dispatcher.Any)

def unsubscribe():
    dispatcher.disconnect(store_message, signal=dispatcher.Any, sender=dispatcher.Any)
    _close_all_handlers()

# ------
# logging to file system for later analysis

def get_file_handler(signal, file_type="pickle"):
    read_type = "ab+" if file_type == "pickle" else "a+"
    if signal not in file_handlers:
        os.makedirs(_path, exist_ok=True)
        file_handlers[signal] = open(os.path.join(_path, "{}.{}".format(signal, file_type)), read_type)
    return file_handlers[signal]

def log_message(signal, msg):
    # determine if protobuf
    if hasattr(msg, "DESCRIPTOR"):
        _log_protobuf_message(signal, msg)
    else:
        _log_normal_message(signal, msg)

def _log_normal_message(signal, msg):
    handler = get_file_handler(signal, file_type="pickle")
    pickle.dump(msg, handler)

def _log_protobuf_message(signal, msg):
    json_ = MessageToJson(msg)
    json_ = json_.replace("\n", "") + "\n"
    handler = get_file_handler(signal, file_type="json")
    handler.write(json_)

def _close_all_handlers():
    global file_handlers
    for h in file_handlers.values():
        h.close()
    file_handlers = {}

