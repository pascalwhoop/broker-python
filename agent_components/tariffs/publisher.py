from pydispatch import dispatcher

import util.config as cfg
from communication.grpc_messages_pb2 import PBTariffSpecification, PBTariffRevoke
from communication.powertac_communication_server import submit_service
from communication.pubsub import signals
from communication.pubsub.PubSubTypes import SignalConsumer
from util import id_generator


import logging
log = logging.getLogger(__name__)

class TariffPublisher(SignalConsumer):
    def __init__(self):
        super().__init__()
        self.clones = {}

    def subscribe(self):
        dispatcher.connect(self.handle_tariff_spec, signals.PB_TARIFF_SPECIFICATION)
        dispatcher.connect(self.handle_tariff_revoke, signals.PB_TARIFF_REVOKE)
        log.info("tariff publisher is listenening")

    def unsubscribe(self):
        dispatcher.disconnect(self.handle_tariff_spec, signals.PB_TARIFF_SPECIFICATION)
        dispatcher.disconnect(self.handle_tariff_revoke, signals.PB_TARIFF_REVOKE)

    def handle_tariff_spec(self, sender, signal: str, msg: PBTariffSpecification):
        """Handling incoming specs. Let's just clone the babies!"""
        #if from our idol
        if msg.broker == cfg.TARIFF_CLONE_COMPETITOR_AGENT:
            #let's clone this
            msg.broker = cfg.ME
            new_id = id_generator.create_id()
            self.clones[msg.id] = new_id
            msg.id = new_id
            #and send it to the server as if it was ours
            submit_service.send_tariff_spec(msg)

    def handle_tariff_revoke(self, sender, signal: str, msg: PBTariffRevoke):
        """if our idol revokes, let's revoke too"""
        if msg.broker == cfg.TARIFF_CLONE_COMPETITOR_AGENT:
            if msg.tariffId in self.clones:
                #have cloned this tariff
                other = msg.tariffId
                msg.broker = cfg.ME
                msg.tariffId = self.clones[other]
                submit_service.send_tariff_revoke(msg)
                del self.clones[other]

