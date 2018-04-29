####################
# SIGNALS
####################

PT_IN_XML            = "PT_IN_XML"            # incoming xml message strings from the server
PT_OUT_XML           = "PT_OUT_XML"           # xml strings destined to the server
PT_IN_STATE_LINE     = "PT_IN_STATE_LINE"     # state lines read and destined for parsing
TICK                 = "TICK"                 # timestep tick
COMP_SUB_EST         = "COMP_SUB_EST"         # subscription estimation by agent component
COMP_USAGE_EST       = "COMP_USAGE_EST"       # usage estimatino by agent component
COMP_WHOLESALE       = "COMP_WHOLESALE"       # wholesale trading action by agent component
COMP_TARIFF_ACTION   = "COMP_TARIFF_ACTION"   # tariff action by agent component
STATE_EXTRACTOR_NEXT = "STATE_EXTRACTOR_NEXT" # sent when the agent requests the next environment parsing
