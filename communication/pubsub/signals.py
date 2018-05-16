# Using PyDispatcher for publish/subscribe activity within the python environment
# http://pydispatcher.sourceforge.net/
####################
# SIGNALS
####################

# These are names for signals that can be subscribed to.
PT_IN_STATE_LINE     = "PT_IN_STATE_LINE"     # state lines read and destined for parsing
COMP_SUB_EST         = "COMP_SUB_EST"         # subscription estimation by agent component
COMP_USAGE_EST       = "COMP_USAGE_EST"       # usage estimatino by agent component
COMP_WHOLESALE       = "COMP_WHOLESALE"       # wholesale trading action by agent component
COMP_TARIFF_ACTION   = "COMP_TARIFF_ACTION"   # tariff action by agent component
STATE_EXTRACTOR_NEXT = "STATE_EXTRACTOR_NEXT" # sent when the agent requests the next environment parsing

# any Protobuf message is a signal that can be subscribed to.
# beware to keep this list up-to-date and using these instead of hard-coding the names in in the listeners
PB_ACTIVATE                 = "PBActivate"
PB_BALANCE_REPORT           = "PBBalanceReport"
PB_BALANCING_CONTROL_EVENT  = "PBBalancingControlEvent"
PB_BALANCING_TRANSACTION    = "PBBalancingTransaction"
PB_BANK_TRANSACTION         = "PBBankTransaction"
PB_CAPACITY_TRANSACTION     = "PBCapacityTransaction"
PB_CASH_POSITION            = "PBCashPosition"
PB_CLEARED_TRADE            = "PBClearedTrade"
PB_COMPETITION              = "PBCompetition"
PB_CUSTOMER_BOOTSTRAP_DATA  = "PBCustomerBootstrapData"
PB_DISTRIBUTION_REPORT      = "PBDistributionReport"
PB_DISTRIBUTION_TRANSACTION = "PBDistributionTransaction"
PB_MARKET_BOOTSTRAP_DATA    = "PBMarketBootstrapData"
PB_MARKET_POSITION          = "PBMarketPosition"
PB_MARKET_TRANSACTION       = "PBMarketTransaction"
PB_ORDER                    = "PBOrder"
PB_ORDERBOOK                = "PBOrderbook"
PB_PROPERTIES               = "PBProperties"
PB_SIM_PAUSE                = "PBSimPause"
PB_SIM_RESUME               = "PBSimResume"
PB_TARIFF_REVOKE            = "PBTariffRevoke"
PB_TARIFF_SPECIFICATION     = "PBTariffSpecification"
PB_TARIFF_STATUS            = "PBTariffStatus"
PB_TARIFF_TRANSACTION       = "PBTariffTransaction"
PB_TIMESLOT_COMPLETE        = "PBTimeslotComplete"
PB_TIMESLOT_UPDATE          = "PBTimeslotUpdate"
PB_WEATHER_FORECAST         = "PBWeatherForecast"
PB_WEATHER_REPORT           = "PBWeatherReport"

# old signals
# PT_IN_XML            = "PT_IN_XML"            # incoming xml message strings from the server
# PT_OUT_XML           = "PT_OUT_XML"           # xml strings destined to the server
