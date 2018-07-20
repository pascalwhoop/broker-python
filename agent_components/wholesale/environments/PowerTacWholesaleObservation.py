from typing import List

import numpy as np

from communication.grpc_messages_pb2 import PBOrderbook, PBClearedTrade, PBMarketTransaction


class PowerTacWholesaleObservation:
    """Helper class that wraps all the components of an observation that can be passed to the agent.
    Each agent implementation can decide what parts of this observation to make use of"""

    def __init__(self, hist_avg_prices: np.array,
                 step: int,
                 orderbooks: List[PBOrderbook],
                 purchases: List[np.array],
                 cleared_trades: List[PBClearedTrade],
                 predictions: List[float],
                 actions: List[np.array],
                 internals = None):
        self.hist_avg_prices: np.array = hist_avg_prices
        self.step = step
        self.orderbooks: List[PBOrderbook] = orderbooks
        self.purchases: List[PBMarketTransaction] = purchases
        self.cleared_trades: List[PBClearedTrade] = cleared_trades
        self.predictions: List[float] = predictions
        self.actions: List[np.array] = actions
        self.internals = internals #used by tensorforce