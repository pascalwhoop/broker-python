import itertools
from collections import deque
from typing import Coroutine, Dict, Generator, List
import util.config as cfg

from pydispatch import dispatcher

import communication.pubsub.signals

import numpy as np
from gym import Env, spaces
from gym.spaces import Box

from agent_components.demand.estimator import CustomerPredictions
from agent_components.wholesale.util import calculate_running_average
from communication.grpc_messages_pb2 import PBMarketTransaction, PBTimeslotUpdate, PBClearedTrade, PBOrderbook
from communication.pubsub.PubSubTypes import SignalConsumer
import communication.pubsub.signals as signals
from util.utils import deprecated


class PowerTacWholesaleAgent:
    """Abstract wholesale agent that can act in a `PowerTacEnv`"""

    def forward(self, observation: "PowerTacWholesaleObservation") -> np.array:
        """Get an action based on an observation."""
        raise NotImplementedError

    def backward(self, action, reward):
        """Receive the reward for the action and learn from historical data"""
        raise NotImplementedError

    def learn(self, observation, action, reward, observation2):
        raise NotImplementedError


class WholesaleEnvironmentManager(SignalConsumer):
    """This class ties together the powertac side of RL and the "classic" side of RL, i.e. the agent driven side.
    Agents in the classic literature step the environment after they have made a decision. PowerTAC doesn't wait for an agent.
    If the agent is too slow, shit goes on. So this ties it all together. """

    def __init__(self, agent:PowerTacWholesaleAgent):
        super().__init__()
        self.environments: Dict[int, "PowerTacEnv"] = {}  # a map of Environments. Key is the target timestep
        self.agent: PowerTacWholesaleAgent = agent

        self.historical_average_prices = {}  # a map of arrays

    def subscribe(self):
        """Subscribe to any newly incoming messages from the server"""
        dispatcher.connect(self.handle_market_transaction, signal=signals.PB_MARKET_TRANSACTION)
        dispatcher.connect(self.handle_timeslot_update, signal=signals.PB_TIMESLOT_UPDATE)
        dispatcher.connect(self.handle_cleared_trade, signal=signals.PB_CLEARED_TRADE)
        dispatcher.connect(self.handle_predictions, signal=signals.COMP_USAGE_EST)
        # TODO
        # tells imbalance after DU has performed balancing
        # dispatcher.connect(None,signal=signals.PB_BALANCE_REPORT)
        # tells the component that the next iteration needs to be calculated.
        # dispatcher.connect(None,signal=signals.PB_TIMESLOT_COMPLETE)

    def unsubscribe(self):
        """unsubscribe from all pubsub messages"""
        dispatcher.disconnect(self.handle_market_transaction, signal=signals.PB_MARKET_TRANSACTION)
        dispatcher.disconnect(self.handle_timeslot_update, signal=signals.PB_TIMESLOT_UPDATE)
        dispatcher.disconnect(self.handle_cleared_trade, signal=signals.PB_CLEARED_TRADE)

    def handle_market_transaction(self, sender, signal: str, msg: PBMarketTransaction):
        if msg.timeslot in self.environments:
            self.environments[msg.timeslot].handle_market_transaction(msg)
        else:
            # this should not happen
            raise Exception("missing environment detected")

    def handle_timeslot_update(self, sender, signal: str, msg: PBTimeslotUpdate):
        """Comes in at every new timestep. Marks a new tick. If no MarketTransactions have come in until now,
        we won't see any for that TS --> no trade"""
        # learn on all "previously active" environments
        for env in self.environments.values():
            env.handle_timeslot_update(msg)
        # then remove just finished ts environment because it's not active anymore
        for ts in list(self.environments.keys()):
            if ts not in range(msg.firstEnabled, msg.lastEnabled + 1):
                del self.environments[ts]

        # create new environments for all newly active
        # pass agent to each environment, they need the agent to interact with it.
        for ts in range(msg.firstEnabled, msg.lastEnabled + 1):
            if ts not in self.environments:
                self.environments[ts] = PowerTacEnv(self.agent, ts, self.get_historical_prices(ts))

    def handle_cleared_trade(self, sender, signal: str, msg: PBClearedTrade):
        self.append_historical(msg)
        if msg.timeslot in self.environments:
            self.environments[msg.timeslot].handle_cleared_trade(msg)
        else:
            raise Exception("missing environment detected")

    def handle_predictions(self, sender, signal: str, msg: List[CustomerPredictions]):
        """msg is an array of predictions for the next 24 ts for a customer. Don't care which customer, just care
        how much we need to buy here."""

        # assumes next x timesteps
        preds = self.get_sums_from_preds(msg)
        for ts in preds:
            self.environments[ts].handle_prediction(preds[ts])

    def get_avg_for_ts(self, ts):
        """gets the average price for the given ts based on the historical clearedTrade data"""
        ts_data = np.array(self.historical_average_prices[ts])
        return calculate_running_average(ts_data)

    def append_historical(self, msg: PBClearedTrade):
        """adds a cleared trade to the historical stats data"""
        ts = msg.timeslot
        if ts not in self.historical_average_prices:
            self.historical_average_prices[ts] = []
        self.historical_average_prices[ts].append([msg.executionMWh, msg.executionPrice])

    def get_historical_prices(self, target_ts):
        start = target_ts - cfg.WHOLESALE_HISTORICAL_DATA_LENGTH
        # minimum timestep is one
        start = start if start >= 1 else 1

        avgs = [calculate_running_average(np.array(self.historical_average_prices[i])) for i in range(start, target_ts)]
        # case A: no data --> just zeros
        if len(avgs) == 0:
            return np.zeros(cfg.WHOLESALE_HISTORICAL_DATA_LENGTH)
        # padding at beginning with the first value of avgs
        # case B: some padding needed
        whole = np.zeros(cfg.WHOLESALE_HISTORICAL_DATA_LENGTH)
        whole.fill(avgs[0])
        whole[-len(avgs):] = avgs
        # "case" C: no padding needed
        return whole

    def get_sums_from_preds(self, preds_list: List[CustomerPredictions]) -> Dict[int, float]:
        """
        Calculates the sum per ts from the list of customerpredictions
        :param preds_list:
        :return:
        """
        # iterate over customers
        ts_preds = {}
        # iterate over customers
        for cust in preds_list:
            cu_preds = cust.predictions
            # iterate over ts of each customer
            for ts in cu_preds:
                if ts not in ts_preds:
                    ts_preds[ts] = 0
                ts_preds[ts] += cu_preds[ts]
        return ts_preds


class PowerTacEnv(Env):
    """A close resembling of the OpenAI Gym but in PowerTAC, the flow of execution is reversed. That means,
    the `step` method is expected to `yield` and be a Generator that data can be passed to.  that gets fed more data whenever it arrives to
    calculate the reward etc.
    """

    def __init__(self, agent: PowerTacWholesaleAgent, target_ts, historical_prices):
        super().__init__()
        # final
        self.agent = agent
        self._target_timeslot = target_ts
        # changing
        self._historical_prices = deque(maxlen=cfg.WHOLESALE_HISTORICAL_DATA_LENGTH)
        self._historical_prices.extend(historical_prices)

        self._step = 0
        self.orderbooks: Dict[int, PBOrderbook] = {}  #
        self.purchases: Dict[int, PBMarketTransaction] = {}
        self.cleared_trades: Dict[int, PBClearedTrade] = {}
        self.actions: Dict[int, np.array] = {}
        self.predictions: List[float] = []

    def step(self, action) -> Generator:
        # TODO crit > implement and test
        raise NotImplementedError

    def reset(self):
        """Doesn't do anything, our envs never get reset, just run once and then dismissed"""
        pass

    def render(self, mode='logging'):
        """nothing to render in powertac"""
        pass

    def close(self):
        raise NotImplementedError

    def handle_timeslot_update(self, msg: PBTimeslotUpdate):
        # TODO crit > implement and test
        raise NotImplementedError

    def handle_prediction(self, prediction):
        """
        at this point the broker is ready to calculate an action! This component is not aware of the exact interests
        of the broker in the environment and therefore it's best to hand the broker everything we know about the environment.
        That usually includes:
        - historical price data of the last X time steps
        - historical price data of the last Y market closings for the target timestep
        - historical predictions for the target timestep
        - historical Orderbooks for the target timestep
        - historical purchases for the target timeslot

        :param prediction: The newest prediction for this target step
        :return:
        """
        self.predictions.append(prediction)
        obs = PowerTacWholesaleObservation(hist_avg_prices=self._historical_prices,
                                           step=self._step,
                                           orderbooks=self.orderbooks,
                                           purchases=self.purchases,
                                           predictions=self.predictions,
                                           cleared_trades=self.cleared_trades,
                                           actions=self.actions
                                           )
        self.agent.forward(obs)
        # TODO crit > implement and test

    def handle_cleared_trade(self, msg: PBClearedTrade):
        # just storing this for later
        self.cleared_trades[msg.timeslot] = msg

    def handle_market_transaction(self, msg: PBMarketTransaction):
        """this tells the agent when it was able to buy something successfully"""
        self.purchases[msg.timeslot] = msg


class PowerTacWholesaleObservation:
    """Helper class that wraps all the components of an observation that can be passed to the agent.
    Each agent implementation can decide what parts of this observation to make use of"""

    def __init__(self, hist_avg_prices: np.array,
                 step: int,
                 orderbooks: Dict[int, PBOrderbook],
                 purchases: Dict[int, np.array],
                 cleared_trades: Dict[int, PBClearedTrade],
                 predictions:List[float],
                 actions: Dict[int, np.array]):
        self.hist_avg_prices: np.array = hist_avg_prices
        self.step = step
        self.orderbooks: Dict[int, PBOrderbook] = orderbooks
        self.purchases: Dict[int, PBMarketTransaction] = purchases
        self.cleared_trades: Dict[int, PBClearedTrade] = cleared_trades
        self.predictions:List[float] = predictions
        self.actions: Dict[int, np.array] = actions


# not really needed anymore
#@deprecated
class WholesaleObservationSpace(spaces.Box):
    """
    - demand prediction - purchases 24x float
    - historical prices of currently traded TS 24x24 float (with diagonal TR-BL zeros in bottom right)
    - historical prices of last 168 timeslots
    - ... TODO more?
    """

    def __init__(self):
        # box needs min and max. using signed int32 min/max
        required_energy = Box(low=cfg.np_low, high=cfg.np_high, shape=(1,), dtype=np.float32)
        historical_prices = Box(low=cfg.np_low, high=cfg.np_high, shape=(168,), dtype=np.float32)
        current_prices = Box(low=cfg.np_low, high=cfg.np_high, shape=(24, 2), dtype=np.float32)
        super().__init__(low=cfg.np_low, high=cfg.np_high, shape=(1 + 24,), dtype=np.float32)


class WholesaleActionSpace(spaces.Box):
    """
    A wholesale action is anywhere in the 2D space with Dim 1 being limited to [-1,+1] and Dim 2 to [-2.0, +2.0]
        - Dimension 0 describes how much mWh to buy. Positive numbers buy, negative numbers sell The network gets fed a prediction regarding its portfolio balance.
          If the prediction is -243 (i.e. missing 243mWh for this timeslot to be matching the predicted demand) then
          - 0.5 means buying 243mWh.
          - 0 means buy nothing,
          - -0.5 means sell another 243mWh and
          - -1 means sell twice the predicted
          imbalance. If the prediction is +230 then 0.5 means buying another 230 and -0.5 means selling 230.
        - Dimension 1 describes the limit price. This is a mapping to the limit price.
          It's based on the known prices for the target timeslot.
             0 --> pay nothing
            -1 --> buy for 2x average known price
            +1 --> sell for 2x average known price
            +0.5 --> sell for average known price
            TODO is this maybe not large enough? increase box size for more freedom in pricing

    """

    def __init__(self):
        a = Box(low=np.array([-1.0, -1.0]), high=np.array([+1.0, +1.0]), dtype=np.float32)
        super().__init__(low=a.low, high=a.high, dtype=np.float32)
