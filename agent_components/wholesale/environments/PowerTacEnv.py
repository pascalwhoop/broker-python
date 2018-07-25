from collections import deque

import logging
import numpy as np
from gym import Env
from pydispatch import dispatcher
from typing import Generator, List

import communication.pubsub.signals as signals
import util.config as cfg
from agent_components.wholesale.environments.PowerTacWholesaleAgent import PowerTacWholesaleAgent
from agent_components.wholesale.environments.PowerTacWholesaleObservation import PowerTacWholesaleObservation
from agent_components.wholesale.util import calculate_running_averages, calculate_balancing_needed
from communication.grpc_messages_pb2 import PBClearedTrade, PBMarketTransaction, PBOrder, \
    PBOrderbook, PBTimeslotUpdate, PBTariffTransaction, PBBalancingTransaction

log = logging.getLogger(__name__)


class PowerTacEnv(Env):
    """A close resembling of the OpenAI Gym but in PowerTAC, the flow of execution is reversed. This means the agent doesn't call the environment but the other way around
    """

    def __init__(self, agent: PowerTacWholesaleAgent, reward_function, target_ts, historical_prices):
        super().__init__()
        # final
        self.internals = []
        self.agent = agent
        self.reward_function = reward_function
        self.realized_usage = 0 #usage in mWh!
        self._target_timeslot = target_ts
        # changing
        self._historical_prices = deque(maxlen=cfg.WHOLESALE_HISTORICAL_DATA_LENGTH)
        self._historical_prices.extend(historical_prices)

        self._step = 0
        self.orderbooks: List[PBOrderbook] = []  #
        self.purchases: List[PBMarketTransaction] = []
        self.balancing_tx: PBBalancingTransaction = None
        self.cleared_trades: List[PBClearedTrade] = []
        self.observations: List[PowerTacWholesaleObservation] = []
        #the functional (powertac logic) actions, i.e. mWh, price
        self.actions: List[np.array] = []
        #the raw action values returned by the NN
        self.nn_actions = []
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
        self._step += 1
        last_action = self.actions[-1] if self.actions else None
        last_observation = self.observations[-1] if self.observations else None
        reward = self.reward_function(self)
        reward *= 1000
        log.debug("reward at TS {} is {}".format(self._step, reward))
        dispatcher.send(signals.COMP_WS_REWARD, msg=reward)
        #crashes because realized_usage is 0 ... why?
        if last_observation is None or last_action is None or self.realized_usage == 0:
            return
        # TODO crit > calculate reward not happening!
        #this doesn't get reached after a few iterations. must be the if above.
        obs_pred = last_observation[24+168:]
        self.agent.backward(self, last_action, reward)
        return reward


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
        log.debug("predictions received in wholesale, starting trading actions")
        self.predictions.append(prediction)
        action, nn_action, internals = self.agent.forward(self)
        self.internals.append(internals)
        # store the action and observation
        self.actions.append(action)
        self.nn_actions.append(nn_action)
        self.send_order(action)

    def handle_cleared_trade(self, msg: PBClearedTrade):
        # just storing this for later
        self.cleared_trades.append(msg)

    def handle_market_transaction(self, msg: PBMarketTransaction):
        """this tells the agent when it was able to buy something successfully"""
        self.purchases.append(msg)

    def send_order(self, action):
        order = PBOrder(broker=cfg.ME, timeslot=self._target_timeslot, mWh=action[0], limitPrice=action[1])
        log.debug("order ! {} mWh, -- {} $".format(order.mWh, order.limitPrice))
        # gotta send via dispatcher
        dispatcher.send(signals.OUT_PB_ORDER, msg=order)

    def handle_tariff_transaction(self, msg:PBTariffTransaction):
        self.realized_usage += msg.kWh / 1000
        pass


    def get_last_known_market_price(self):
        """returns the best guess at the latest "known price" i.e. what was the latest traded price for the target slot"""
        #if already traded, last trade clearing
        if self.cleared_trades:
            return self.cleared_trades[-1].executionPrice
        # if just historicals, latest TS before the target slot
        if self._historical_prices:
            return self._historical_prices[-1]
        #else 0
        return 0

    def handle_balancing_transaction(self, msg: PBBalancingTransaction):
        """The DU balancing. For all intents of this component, it's just another transaction. But we'll store it as a separate value in the object"""
        sum_power_flow = np.array([p.mWh for p in self.purchases]).sum()
        bal_should = self.realized_usage + sum_power_flow
        self.balancing_tx = msg
        if abs(bal_should +  msg.kWh / 1000 * -1) > 0.00001:
            print(" bal {} --- bought {}".format(msg.kWh / 1000 * -1, sum_power_flow))
            raise ValueError("balancing_tx not correct")



