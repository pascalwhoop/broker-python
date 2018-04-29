import logging
LOGGER = logging.getLogger(__name__)

class WholesaleStore(object):

    def __init__(self, env):
        self.env                 = env
        self.orderbooks          = {}
        self.market_transactions = {}
        self.market_positions    = {}
        self.cleared_trades      = {}

        
