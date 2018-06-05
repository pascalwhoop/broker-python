import logging

import numpy as np
from keras import Model
from pydispatch import dispatcher
from sklearn.preprocessing import MinMaxScaler

import communication.pubsub.signals as signals
import util.config as cfg
from agent_components.demand.data import sequence_for_usages
from agent_components.demand.learning.dense_v2.learner import DenseLearner
from communication.grpc_messages_pb2 import PBCustomerBootstrapData, PBSimEnd, PBTariffTransaction, PBTimeslotComplete, \
    PBTxType
from util.learning_utils import reload_model_customer_nn, store_model_customer_nn

log = logging.getLogger(__name__)



class Estimator:
    """
    Central class that exposes an API to the rest of the broker to get estimations for customers. It automatically subscribes
    on all interesting events and learns whenever new information arrives.

    """

    def __init__(self):
        self.dl = DenseLearner('estimator', True)
        self.models = {}  # models can be looked up via customer name
        self.scalers = {}  # scalers can also be looked up via customer name. They scale the customer data
        self.usages = {}  # map of maps. first map --> customer_name, second map --> timeslot ID
        self.updated = set()
        self.current_timeslot = 0
        self.predictions = {}  # customers -> timeslots -> 24x predictions

        #subscribing to messages
        self.subscribe()

    def subscribe(self):
        """Subscribes this object to the events of interest to the estimator"""
        dispatcher.connect(self.handle_tariff_transaction_event, signals.PB_TARIFF_TRANSACTION)
        dispatcher.connect(self.handle_customer_bootstrap_data_event, signals.PB_CUSTOMER_BOOTSTRAP_DATA)
        dispatcher.connect(self.handle_timeslot_complete, signals.PB_TIMESLOT_COMPLETE)
        dispatcher.connect(self.handle_sim_end, signals.PB_SIM_END)

    def unsubscribe(self):
        dispatcher.disconnect(self.handle_tariff_transaction_event, signals.PB_TARIFF_TRANSACTION)
        dispatcher.disconnect(self.handle_customer_bootstrap_data_event, signals.PB_CUSTOMER_BOOTSTRAP_DATA)
        dispatcher.disconnect(self.handle_timeslot_complete, signals.PB_TIMESLOT_COMPLETE)
        dispatcher.disconnect(self.handle_sim_end, signals.PB_SIM_END)

    def handle_tariff_transaction_event(self, sender, signal: str, msg: PBTariffTransaction):
        """Add any consume/produce to historic records for customer"""
        if msg.txType is PBTxType.Value("CONSUME") or msg.txType is PBTxType.Value("PRODUCE"):
            self.add_transaction(msg)
        else:
            # not of interest to this component
            pass

    def handle_customer_bootstrap_data_event(self, sender, signal: str, msg: PBCustomerBootstrapData):
        # apply all bootstrap data
        name = msg.customerName
        for i, r in enumerate(msg.netUsage):
            self._apply_usage(name, r, i + 1)
        X = np.array(list(self.usages[name].values()))
        # use it to create a scaler
        scaler = MinMaxScaler()
        scaler.fit(X.reshape(-1, 1))
        self.scalers[name] = scaler

        # scale data before using it to learn
        X_scaled = scaler.transform(X.reshape(-1, 1)).flatten()
        seq = sequence_for_usages(X_scaled, True)

        # get model and learn with 10 epochs
        model = self.get_model(name)

        log.info("fitting model for customer {}".format(name))
        model.fit_generator(seq, epochs=1, verbose=1, use_multiprocessing=False)

    def handle_timeslot_complete(self, sender, signal: str, msg: PBTimeslotComplete):
        self.current_timeslot = msg.timeslotIndex + 1
        # trigger learning on all customers for recently completed TS
        self.process_customer_new_data()

    def handle_sim_end(self, sender, signal: str, msg: PBSimEnd):
        for m in self.models.items():
            store_model_customer_nn(m[1], m[0], self.dl.model_name)

        # remove all data
        self.usages = {}
        self.predictions = {}
        self.current_timeslot = 0

    def add_transaction(self, tx: PBTariffTransaction):
        customer_name = tx.customerInfo.name
        kwh = tx.kWh
        timeslot = tx.postedTimeslot
        self._apply_usage(customer_name, kwh, timeslot)

    def _apply_usage(self, customer_name, kwh, timeslot):
        if customer_name not in self.usages:
            self.usages[customer_name] = {}
        if timeslot not in self.usages[customer_name]:
            self.usages[customer_name][timeslot] = 0
        self.usages[customer_name][timeslot] += kwh

    def get_model(self, customer_name) -> Model:
        """Gets a model for the given customer name from the local models dict or creates a new one if none exists"""
        if customer_name not in self.models:
            #trying to reload from fs first
            model = reload_model_customer_nn(customer_name, self.dl.model_name)
            if model is None:
                log.info("getting new model for {}".format(customer_name))
                model = self.dl.fresh_model()
            self.models[customer_name] = model
        return self.models[customer_name]

    def process_customer_new_data(self):
        """after the timeslot is completed, this triggers prediction and learning on all timeslots."""
        for c in self.usages.items():
            #scale the data
            backw_size = cfg.DEMAND_ONE_WEEK + cfg.DEMAND_FORECAST_DISTANCE + 10  # equivalent as learning of each info 10x
            usages = np.array(list(c[1].values()))[-backw_size:]
            scaler = self.scalers[c[0]]
            usages_scaled = scaler.transform(usages.reshape(-1,1)).flatten()
            # first, learn from the newly available knowledge
            seq = sequence_for_usages(usages_scaled[-backw_size:], True)
            model = self.get_model(c[0])
            model.fit_generator(seq)

            #then predict the next 24h
            predictions_scaled = model.predict(usages_scaled[-cfg.DEMAND_ONE_WEEK:])
            #predictions for the next 24h timesteps
            predictions = scaler.inverse_transform(predictions_scaled)
            self.store_predictions(c[0], predictions)

            #and publish the new prediction to anyone who is interested
            self.publish_predictions(c[0], predictions, self.current_timeslot)

    def publish_predictions(self, customer_name, predictions, first_ts):
        pred = CustomerPredictions(customer_name, predictions, first_ts)
        dispatcher.send(signals.COMP_USAGE_EST, msg=pred)

    def store_predictions(self, customer_name: str, predictions: np.array):
        """Stores all new predictions in the memory. This let's us compare predictions and real values later. """
        for i, ts in enumerate(range(self.current_timeslot+1, self.current_timeslot+1+24)):
            if customer_name not in self.predictions:
                self.predictions[customer_name] = {}
            if ts not in self.predictions[customer_name]:
                self.predictions[customer_name][ts] = []
            self.predictions[customer_name][ts].append(predictions[i])



# hacky solution right now. Trying to get results


class CustomerPredictions:
    """Holds a 24 hour set of predictions"""
    def __init__(self, name, predictions, first_ts):
        self.customer_name = name
        tss = [i for i in range(first_ts, first_ts + len(predictions))]
        self.predictions = {}
        for i, ts in enumerate(tss):
            self.predictions[ts] = predictions[i]
