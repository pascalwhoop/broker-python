from pydispatch import dispatcher

import communication.pubsub.signals as signals
from communication.grpc_messages_pb2 import PBTariffTransaction


class Estimator:
    """
    Central class that exposes an API to the rest of the broker to get estimations for customers. It automatically subscribes
    on all interesting events and learns whenever new information arrives.

    """

    def __init__(self):
        self.subscribe()

    def subscribe(self):
        """Subscribes this object to the events of interest to the estimator"""
        dispatcher.connect(self.handle_tariff_transaction_event, signals.PB_TARIFF_TRANSACTION)
        dispatcher.connect(self.handle_customer_bootstrap_data_event, signals.PB_CUSTOMER_BOOTSTRAP_DATA)

    def handle_tariff_transaction_event(self, sender, signal: str, message: PBTariffTransaction):
        pass

    def handle_customer_bootstrap_data_event(self, sender, signal:str, message: PBTariffTransaction):
        pass
