import logging
from typing import List, Dict
import tensorflow as tf

import numpy as np
from pydispatch import dispatcher
from sklearn.preprocessing import MinMaxScaler

import communication.pubsub.signals as signals
import util.config as cfg
from agent_components.demand.learning.data import sequence_for_usages
from communication.grpc_messages_pb2 import PBCustomerBootstrapData, PBSimEnd, PBTariffTransaction, PBTimeslotComplete, \
    PBTxType
from communication.pubsub.SignalConsumer import SignalConsumer

log = logging.getLogger(__name__)



class Estimator(SignalConsumer):
    """
    Central class that exposes an API to the rest of the broker to get estimations for customers. It automatically subscribes
    on all interesting events and learns whenever new information arrives.

    """

    def __init__(self, model):
        super().__init__()
        self.graph = tf.get_default_graph() #required to do multi threaded tensorflow actions : https://github.com/keras-team/keras/issues/2397
        self.model = model
        self.scalers = {}  # scalers can also be looked up via customer name. They scale the customer data
        self.usages: Dict[int, Dict[int,float]] = {}  # map of maps. first map --> customer_name, second map --> timeslot ID
        self.customer_counts = {} #map that stores the number of customers per customer_name
        self.customer_populations = {}
        self.updated = set()
        self.current_timeslot = 0
        self.predictions = {}  # customers -> timeslots -> 24x predictions

    def subscribe(self):
        """Subscribes this object to the events of interest to the estimator"""
        dispatcher.connect(self.handle_tariff_transaction_event, signals.PB_TARIFF_TRANSACTION)
        dispatcher.connect(self.handle_timeslot_complete, signals.PB_TIMESLOT_COMPLETE)
        dispatcher.connect(self.handle_sim_end, signals.PB_SIM_END)
        dispatcher.connect(self.handle_customer_bootstrap_data_event, signals.PB_CUSTOMER_BOOTSTRAP_DATA)
        log.info("estimator is listening!")

    def unsubscribe(self):
        dispatcher.disconnect(self.handle_tariff_transaction_event, signals.PB_TARIFF_TRANSACTION)
        dispatcher.disconnect(self.handle_timeslot_complete, signals.PB_TIMESLOT_COMPLETE)
        dispatcher.disconnect(self.handle_sim_end, signals.PB_SIM_END)
        dispatcher.disconnect(self.handle_customer_bootstrap_data_event, signals.PB_CUSTOMER_BOOTSTRAP_DATA)

    def handle_tariff_transaction_event(self, sender, signal: str, msg: PBTariffTransaction):
        """watch consume/produce and tariff subscriptions"""
        if msg.txType is PBTxType.Value("CONSUME") or msg.txType is PBTxType.Value("PRODUCE"):
            if msg.regulation is False:
                self.handle_usage(msg)
        #keep track of our customers
        if msg.txType is PBTxType.Value("SIGNUP") or msg.txType is PBTxType.Value("WITHDRAW"):
            self.handle_customer_change(msg)
        else:
            # not of interest to this component
            pass

    def handle_customer_change(self, msg: PBTariffTransaction):
        """
        whenever a SIGNUP or WITHDRAW happens, we need to adapt the customer counts in the estimator
        :param msg:
        :return:
        """

        customer = msg.customerInfo.name
        c_count = msg.customerCount

        #remember the population
        self.customer_populations[customer] = msg.customerInfo.population

        if msg.txType is PBTxType.Value("WITHDRAW"):
            c_count *= -1
        if msg.customerInfo.name not in self.customer_counts:
            self.customer_counts[customer] = 0

        self.customer_counts[customer] += c_count

        if self.customer_counts[customer] == 0:
            del self.customer_counts[customer]

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
        """Triggers an estimation round for all customers"""
        self.current_timeslot = msg.timeslotIndex + 1
        # trigger learning on all customers for recently completed TS
        self.process_customer_new_data()

    def handle_sim_end(self, sender, signal: str, msg: PBSimEnd):
        # remove all data
        self.usages = {}
        self.predictions = {}
        self.current_timeslot = 0

    def handle_usage(self, tx: PBTariffTransaction):
        """Every new usage that is given to the estimator is handled here. It's first scaled to the population of the
        customer and then stored """
        customer_name = tx.customerInfo.name
        kwh = self._convert_to_whole_population(tx.kWh, tx.customerInfo.name)
        timeslot = tx.postedTimeslot
        self._apply_usage(customer_name, kwh, timeslot)

    def _convert_to_whole_population(self, usage, name):
        """pass in the usage that was summed up for this timeslot. It'll be scaled up to the whole population of the
        customer to estimate on """
        part = self.customer_counts[name] / self.customer_populations[name]
        usage = usage / part
        return usage

    def _convert_from_whole_population(self, customer_prediction: "CustomerPredictions"):
        """Fixes the customerPredictions object to be appropriate for the actual number of ppl subscribed to us"""
        for p in customer_prediction.predictions:
            usage = customer_prediction.predictions[p]
            name = customer_prediction.customer_name
            part = self.customer_counts[name] / self.customer_populations[name]
            usage = usage * part
            customer_prediction.predictions[p] = usage
        return customer_prediction


    def _apply_usage(self, customer_name, kwh, timeslot):
        if customer_name not in self.usages:
            self.usages[customer_name] = {}
        if timeslot not in self.usages[customer_name]:
            self.usages[customer_name][timeslot] = 0
        self.usages[customer_name][timeslot] += kwh


    def process_customer_new_data(self):
        """after the timeslot is completed, this triggers prediction and learning on all timeslots."""

        log.info("starting prcessing of customer data after round")

        now = self.current_timeslot
        #the 24  TS BEFORE now are         TARGETS
        #the 168 TS BEFORE the targets are INPUT
        step1 = cfg.DEMAND_FORECAST_DISTANCE
        step2 = step1 + cfg.DEMAND_ONE_WEEK
        target_ts = np.arange(now-step1,now)
        input_ts = np.arange(now-step2, now-step1)
        pred_input_ts = np.arange(now-cfg.DEMAND_ONE_WEEK, now)
        self._ensure_all_there(np.arange(now - step2, now))

        # iterate over all customers
        #make batches
        X_ALL = []
        Y_ALL = []
        X_PRED_ALL = []
        scalers = []
        cust_row_map = []

        #make data into batches that can be passed to the NN
        current_customers_data = {customer: self.usages[customer] for customer in self.usages if customer in self.customer_counts}
        log.info("predicting usage for {} customers".format(len(list(current_customers_data.values()))))
        for c in current_customers_data.items():
            scaler = self.scalers[c[0]]
            #store order of customers
            cust_row_map.append(c[0])
            scalers.append(scaler)
            X = np.array([c[1][i] for i in input_ts])
            X = scaler.transform(X.reshape(-1,1)).flatten()
            X_ALL.append(X)
            Y = np.array([c[1][i] for i in target_ts])
            Y = scaler.transform(Y.reshape(-1,1)).flatten()
            Y_ALL.append(Y)
            X_PRED = np.array([c[1][i] for i in pred_input_ts])
            X_PRED = scaler.transform(X_PRED.reshape(-1,1)).flatten()
            X_PRED_ALL.append(X_PRED)

        X_ALL = np.array(X_ALL)
        Y_ALL = np.array(Y_ALL)
        X_PRED_ALL = np.array(X_PRED_ALL)

        predictions_list:List[CustomerPredictions]= []
        #if no customers subscribed yet
        if len(X_ALL) == 0 or len(Y_ALL) == 0 or len(X_PRED_ALL) == 0:
            dispatcher.send(signal=signals.COMP_USAGE_EST, msg=predictions_list)
        else:
            #predict on all in batch
            with self.graph.as_default():
                preds = self.model.predict(X_PRED_ALL)
            preds = [scalers[i].inverse_transform(d.reshape(-1,1)).flatten() for i,d in enumerate(preds)]
            # and storing / unpacking all batched predictions
            for i, p in enumerate(preds):
                p = p / 1000
                obj_ = CustomerPredictions(name=cust_row_map[i], predictions=p, first_ts=now)
                obj_ = self._convert_from_whole_population(obj_)
                self.store_predictions(obj_.customer_name, p)
                predictions_list.append(obj_)

            for i in range(cfg.DEMAND_FORECAST_DISTANCE):
                log.info("Usage prediced: TIMESLOT {} -- USAGE {}".format(now+i,np.array(preds)[:,i].sum()))

            #after all customers have been predicted, one message is spread
            dispatcher.send(signal=signals.COMP_USAGE_EST, msg=predictions_list)

            #learn on all in batch
            log.info("starting learning on customer data")
            with self.graph.as_default():
                self.model.fit(X_ALL, Y_ALL)
            log.info("learning completed")


    def store_predictions(self, customer_name: str, predictions: np.array):
        """Stores all new predictions in the memory. This let's us compare predictions and real values later."""
        for i, ts in enumerate(range(self.current_timeslot+1, self.current_timeslot+1+24)):
            if customer_name not in self.predictions:
                self.predictions[customer_name] = {}
            if ts not in self.predictions[customer_name]:
                self.predictions[customer_name][ts] = []
            self.predictions[customer_name][ts].append(predictions[i])

    def _ensure_all_there(self, tss:np.array):
        """Ensures that all the usages are recorded for all customers for the given tss"""
        for c in self.usages:
            there = np.array(list(self.usages[c].keys()))
            mask = np.in1d(tss, there, invert=True)
            missing = tss[mask]
            for ts in missing:
                #any missing set to 0
                self.usages[c][ts] = 0





class CustomerPredictions:
    """Holds a 24 hour set of predictions"""
    def __init__(self, name, predictions, first_ts):
        self.customer_name = name
        self.first_ts = first_ts
        tss = [i for i in range(first_ts, first_ts + len(predictions))]

        #predictions. Set them in mWh here, even though PBTariffTransaction reports them in kWh
        self.predictions:Dict[int, float] = {}
        for i, ts in enumerate(tss):
            self.predictions[ts] = predictions[i]


