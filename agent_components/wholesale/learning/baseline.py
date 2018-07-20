import logging
import numpy as np

from agent_components.wholesale.environments.PowerTacEnv import PowerTacEnv
from agent_components.wholesale.environments.PowerTacWholesaleAgent import PowerTacWholesaleAgent
from communication.grpc_messages_pb2 import PBOrderbook
from util.learning_utils import get_tb_cb

log = logging.getLogger(__name__)

model_name = "baseline-log-rl"
tag = ""

def get_instance(tag_, fresh):
    global tag
    tag = tag_
    return BaselineTrader()


class BaselineTrader(PowerTacWholesaleAgent):
    """This baseline trader does exactly what the prediction says. It takes the prediction, offers any price
    (10x the previous, so a lot) and tries to always balance the portfolio no matter what. It adapts the Keras-RL API
    not because I am using Keras here but because the other agents will too and this will talk to the same APIs and behave
    the same way as an NN based agents (except that it is just really stupid). """

    def __init__(self):
        super().__init__('baseline')


    def forward(self, observation: PowerTacEnv) -> np.array:
        """Takes the observation and returns the action that matches it"""

        mWh = self.determine_mWh(observation)
        price = self.determine_price(observation, mWh)

        return np.array([mWh, price]), None, None #passing 3 things back to avoid unpack errors

    def determine_mWh(self, observation):
        # we need to buy the opposite of the customers predictions
        mWh = observation.predictions[-1]
        # but reduce it by what we already purchased
        bought = np.array([a.mWh for a in observation.purchases]).sum()
        mWh = mWh + bought
        mWh *= -1
        return mWh

    def determine_price(self, observation:PowerTacEnv, needed):
        if len(observation.orderbooks) > 0:
            ob: PBOrderbook = observation.orderbooks[-1]
            price = ob.clearingPrice * 10
        else:
            price = observation._historical_prices[-24]
         #if we buy mWh --> negative price
        if needed > 0:
            price = abs(price) * -1 * 10 # offering 10 x the price
        #else positive price
        else:
            price = abs(price) / 10  # asking 1/10th the market price

        return price

    def backward(self, env: PowerTacEnv, action, reward):
        """Does nothing really. """
        pass

    def save_model(self):
        log.info("the baseline agent doesn't need to save itself")
        pass





    def make_logger_callback(self):
        return get_tb_cb("{}_{}".format(model_name, tag), write_graph=False)
