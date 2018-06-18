"""This variant of the communication relies on the idea that the python code is a "server" and the java bridge broker
is a client. This has some advantages: - the majority of message types are sent server-> client - the rpc can be
called on each received message of the server instead of having to have a stream of messages - the messages that are
sent to the server however are now a stream "to the client" (which means to the bridge and then JMS to the server) -
the moment the bridge gets started, the competition starts. This means this part can be started ahead of time and the
classic "start your clients" approach stays the same. The client starts, connects to this and to the other server and
bridges.

This assumes that the server will NOT implement this GRPC approach which was the state of the last call with John in
April '18 """
import asyncio
import functools
import inspect
import logging
import threading
import time
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from pydispatch import dispatcher
from queue import Queue

import grpc
from grpc import _server

import communication.grpc_messages_pb2 as ptac_pb2
from communication.pubsub import signals
from communication.pubsub.PubSubTypes import SignalConsumer
import communication.grpc_messages_pb2_grpc as ptac_grpc
import util.config as cfg
from communication.pubsub.grpc_adapter import publish_pb_message
from util import id_generator
from util.strings import GRPC_SERVER_STARTING

log = logging.getLogger(__name__)
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
loop = asyncio.get_event_loop()

# Server Bootstrap
################################################################################


def serve():
    """Starts the grpc listener on the local machine"""
    server = grpc.server(ThreadPoolExecutor(max_workers=16), interceptors=[CallLogInterceptor()])
    ptac_grpc.add_ContextManagerServiceServicer_to_server(ContextManagerService(), server)
    ptac_grpc.add_MarketManagerServiceServicer_to_server(MarketManagerService(), server)
    ptac_grpc.add_PortfolioManagerServiceServicer_to_server(PortfolioManagerService(), server)
    ptac_grpc.add_ConnectionServiceServicer_to_server(ConnectionService(), server)
    ptac_grpc.add_ExtraSpyMessageManagerServiceServicer_to_server(ExtraSpyMessageManagerService(), server)
    ptac_grpc.add_GameServiceServicer_to_server(GameService(), server)
    # for sending to ptac server
    ptac_grpc.add_SubmitServiceServicer_to_server(submit_service, server)

    address = 'localhost:{}'.format(cfg.GRPC_PORT)
    log.info(GRPC_SERVER_STARTING.format(address))
    server.add_insecure_port(address)
    server.start()
    return server


# Submit methods. This is where the agent can send messages to the server
################################################################################


class SubmitService(ptac_grpc.SubmitServiceServicer, SignalConsumer):
    def __init__(self):
        self._order_queue = Queue()
        self._tariff_spec_queue = Queue()
        self._tariff_revoke_queue = Queue()

    def subscribe(self):
        dispatcher.connect(self.send_order, signal=signals.OUT_PB_ORDER)
        log.info("submitService is listenening")

    def unsubscribe(self):
        dispatcher.disconnect(self.send_order, signal=signals.OUT_PB_ORDER)

    def send_order(self, msg: ptac_pb2.PBOrder):
        self._order_queue.put_nowait(msg)

    def send_tariff_revoke(self, msg: ptac_pb2.PBOrder):
        self._order_queue.put_nowait(msg)

    def send_tariff_spec(self, msg: ptac_pb2.PBTariffSpecification):
        self._tariff_spec_queue.put_nowait(msg)

    def submitOrder(self, request, context):
        """DO NOT CALL from python. This is the API to the adapter"""
        while True:
            it = self._order_queue.get()
            yield it

    def submitTariffRevoke(self, request, context):
        """DO NOT CALL from python. This is the API to the adapter"""
        while True:
            it = self._tariff_revoke_queue.get()
            yield it

    def submitTariffSpec(self, request, context):
        """DO NOT CALL from python. This is the API to the adapter"""
        while True:
            it = self._tariff_spec_queue.get()
            yield it


# global submit instance
submit_service = SubmitService()


# Handler Methods. Three classes for the corresponding Java Classes
################################################################################

class ContextManagerService(ptac_grpc.ContextManagerServiceServicer):
    def handlePBBankTransaction(self, request, context):
        """in java, these are overloaded with different class types.
        GRPC doesn't allow same names so we name them as so
        handle<messagetype>()
        """

        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBCashPosition(self, request, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBDistributionReport(self, request, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBCompetition(self, request, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBProperties(self, request, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()


class MarketManagerService(ptac_grpc.MarketManagerServiceServicer):
    def handlePBActivate(self, request, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBCompetition(self, request, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBBalancingTransaction(self, request, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBClearedTrade(self, request: ptac_pb2.PBClearedTrade, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBMarketPosition(self, request: ptac_pb2.PBMarketPosition, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBMarketTransaction(self, request: ptac_pb2.PBMarketTransaction, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBOrderbook(self, request: ptac_pb2.PBOrderbook, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBDistributionTransaction(self, request, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBCapacityTransaction(self, request, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBMarketBootstrapData(self, request, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBWeatherForecast(self, request: ptac_pb2.PBWeatherForecast, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBWeatherReport(self, request: ptac_pb2.PBWeatherReport, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBBalanceReport(self, request, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()


class PortfolioManagerService(ptac_grpc.PortfolioManagerServiceServicer):
    def handlePBCustomerBootstrapData(self, request, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBTariffSpecification(self, request, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBTariffStatus(self, request, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBTariffTransaction(self, request, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBTariffRevoke(self, request, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBBalancingControlEvent(self, request, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()


class ConnectionService(ptac_grpc.ConnectionServiceServicer):
    def pingpong(self, request, context):
        log.info("ping received")
        return ptac_pb2.Empty()


class GameService(ptac_grpc.GameServiceServicer):
    def handlePBSimPause(self, request, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBTimeslotComplete(self, request, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBSimResume(self, request, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBTimeslotUpdate(self, request, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBSimEnd(self, request, context):
        publish_pb_message(request, loop)
        return ptac_pb2.Empty()

    def handlePBBrokerAccept(self, request: ptac_pb2.PBBrokerAccept, context):
        #ignoring the key, we only use the prefix because the key is added in java to the xml string
        id_generator.set_prefix(request.prefix)
        return ptac_pb2.Empty()


class ExtraSpyMessageManagerService(ptac_grpc.ExtraSpyMessageManagerServiceServicer):
    def handlePBOrder(self, request, context):
        # log.info('received a spied upon order message')
        return ptac_pb2.Empty()


# Helper methods
################################################################################

class CallLogInterceptor(grpc.ServerInterceptor):

    def __init__(self):
        super().__init__()
        log.info("logging interceptor loaded")

    def intercept_service(self, continuation, handler_call_details):
        log.debug("call received")
        return continuation(handler_call_details)


