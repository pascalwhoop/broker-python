import logging
from typing import List, Dict

import numpy as np
from pydispatch import dispatcher
from sklearn.preprocessing import MinMaxScaler

import communication.pubsub.signals as signals
import util.config as cfg
from agent_components.demand.learning.data import sequence_for_usages
from communication.grpc_messages_pb2 import PBCustomerBootstrapData, PBSimEnd, PBTariffTransaction, PBTimeslotComplete, \
    PBTxType
from communication.pubsub.PubSubTypes import SignalConsumer

log = logging.getLogger(__name__)



class Estimator(SignalConsumer):
    """
    Central class that exposes an API to the rest of the broker to get estimations for customers. It automatically subscribes
    on all interesting events and learns whenever new information arrives.

    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.scalers = {}  # scalers can also be looked up via customer name. They scale the customer data
        self.usages = {}  # map of maps. first map --> customer_name, second map --> timeslot ID
        self.updated = set()
        self.current_timeslot = 0
        self.predictions = {}  # customers -> timeslots -> 24x predictions

    def subscribe(self):
        """Subscribes this object to the events of interest to the estimator"""
        dispatcher.connect(self.handle_tariff_transaction_event, signals.PB_TARIFF_TRANSACTION)
        dispatcher.connect(self.handle_timeslot_complete, signals.PB_TIMESLOT_COMPLETE)
        dispatcher.connect(self.handle_sim_end, signals.PB_SIM_END)
        dispatcher.connect(self.handle_customer_bootstrap_data_event, signals.PB_CUSTOMER_BOOTSTRAP_DATA)

    def unsubscribe(self):
        dispatcher.disconnect(self.handle_tariff_transaction_event, signals.PB_TARIFF_TRANSACTION)
        dispatcher.disconnect(self.handle_timeslot_complete, signals.PB_TIMESLOT_COMPLETE)
        dispatcher.disconnect(self.handle_sim_end, signals.PB_SIM_END)
        dispatcher.disconnect(self.handle_customer_bootstrap_data_event, signals.PB_CUSTOMER_BOOTSTRAP_DATA)

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

        # TODO skipping bootstrap data, too much to handle in under 5 sec
        # ignoring bootstrap fitting, takes too long otherwise
        return

        # scale data before using it to learn
        X_scaled = scaler.transform(X.reshape(-1, 1)).flatten()
        seq = sequence_for_usages(X_scaled, True)

        #TODO not yet shuffled... maybe I should shuffle this
        log.info("fitting model for customer {}".format(name))
        try:
            self.model.fit_generator(seq, epochs=1, verbose=0, use_multiprocessing=False)
        except Exception as e:
            log.error(e)

    def handle_timeslot_complete(self, sender, signal: str, msg: PBTimeslotComplete):
        self.current_timeslot = msg.timeslotIndex + 1
        # trigger learning on all customers for recently completed TS
        self.process_customer_new_data()

    def handle_sim_end(self, sender, signal: str, msg: PBSimEnd):
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


    def process_customer_new_data(self):
        """after the timeslot is completed, this triggers prediction and learning on all timeslots."""
        predictions_list:List[CustomerPredictions]= []
        for c in self.usages.items():
            #scale the data
            backw_size = cfg.DEMAND_ONE_WEEK + cfg.DEMAND_FORECAST_DISTANCE
            usages = np.array(list(c[1].values()))[-backw_size:]
            scaler = self.scalers[c[0]]
            usages_scaled = scaler.transform(usages.reshape(-1,1)).flatten()
            #first predict the next 24h
            predictions_scaled = self.model.predict(usages_scaled[-cfg.DEMAND_ONE_WEEK:])
            #predictions for the next 24h timesteps
            predictions = scaler.inverse_transform(predictions_scaled)
            self.store_predictions(c[0], predictions)

            #and publish the new prediction to anyone who is interested
            pred = CustomerPredictions(c[0], predictions, self.current_timeslot)
            predictions_list.append(pred)

            # then, learn from the newly available knowledge
            self.model.fit(usages_scaled[:cfg.DEMAND_ONE_WEEK], usages_scaled[-cfg.DEMAND_FORECAST_DISTANCE:])

        #after all customers have been predicted, one message is spread
        dispatcher.send(signal=signals.COMP_USAGE_EST, msg=predictions_list)


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
        self.first_ts = first_ts
        tss = [i for i in range(first_ts, first_ts + len(predictions))]
        self.predictions:Dict[int, float] = {}
        for i, ts in enumerate(tss):
            self.predictions[ts] = predictions[i]
