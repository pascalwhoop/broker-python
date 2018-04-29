"""This variant of the communication relies on the idea that the python code is a "server" and the java bridge broker
is a client. This has some advantages: - the majority of message types are sent server-> client - the rpc can be
called on each received message of the server instead of having to have a stream of messages - the messages that are
sent to the server however are now a stream "to the client" (which means to the bridge and then JMS to the server) -
the moment the bridge gets started, the competition starts. This means this part can be started ahead of time and the
classic "start your clients" approach stays the same. The client starts, connects to this and to the other server and
bridges.

This assumes that the server will NOT implement this GRPC approach which was the state of the last call with John in
April '18 """
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from google.protobuf.json_format import MessageToJson

import grpc
import sys

from env import environment
import communication.grpc_messages_pb2 as ptac_pb2
import communication.grpc_messages_pb2_grpc as ptac_grpc
import util.config as cfg
from util.strings import GRPC_METHOD_NOT_IMPLEMENTED, GRPC_SERVER_STARTING

log = logging.getLogger(__name__)
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

_env = None

# Server Bootstrap
################################################################################

def serve():
    """Starts the grpc listener on the local machine"""
    server = grpc.server(ThreadPoolExecutor(max_workers=1), interceptors=[CallLogInterceptor()])
    ptac_grpc.add_ContextManagerServiceServicer_to_server(ContextManagerService(),     server)
    ptac_grpc.add_MarketManagerServiceServicer_to_server(MarketManagerService(),       server)
    ptac_grpc.add_PortfolioManagerServiceServicer_to_server(PortfolioManagerService(), server)
    ptac_grpc.add_ConnectionServiceServicer_to_server(ConnectionService(), server)

    global _env
    _env = environment.get_instance()


    address = 'localhost:{}'.format(cfg.GRPC_PORT)
    log.info(GRPC_SERVER_STARTING.format(address))
    server.add_insecure_port(address)
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


# Handler Methods. Three classes for the corresponding Java Classes
################################################################################

class ContextManagerService(ptac_grpc.ContextManagerServiceServicer):
    def handlePBBankTransaction(self, request, context):
        """in java, these are overloaded with different class types.
        GRPC doesn't allow same names so we name them as so
        handle<messagetype>()
        """

        warn_about_grpc_not_implemented()
        return ptac_pb2.Empty()

    def handlePBCashPosition(self, request, context):
        warn_about_grpc_not_implemented()
        return ptac_pb2.Empty()

    def handlePBDistributionReport(self, request, context):
        warn_about_grpc_not_implemented()
        return ptac_pb2.Empty()

    def handlePBCompetition(self, request, context):

        warn_about_grpc_not_implemented()
        return ptac_pb2.Empty()

    def handlePBProperties(self, request, context):
        warn_about_grpc_not_implemented()
        return ptac_pb2.Empty()


class MarketManagerService(ptac_grpc.MarketManagerServiceServicer):
    def handlePBActivate(self, request, context):
        warn_about_grpc_not_implemented()
        return ptac_pb2.Empty()

    def handlePBCompetition(self, request, context):
        warn_about_grpc_not_implemented()
        return ptac_pb2.Empty()

    def handlePBBalancingTransaction(self, request, context):
        warn_about_grpc_not_implemented()
        return ptac_pb2.Empty()

    def handlePBClearedTrade(self, request: ptac_pb2.PBClearedTrade, context):
        _env.wholesale_store.cleared_trades[request.id] = request
        return ptac_pb2.Empty()

    def handlePBMarketPosition(self, request: ptac_pb2.PBMarketPosition, context):
        _env.wholesale_store.market_positions[request.id] = request
        return ptac_pb2.Empty()

    def handlePBMarketTransaction(self, request: ptac_pb2.PBMarketTransaction, context):
        _env.wholesale_store.market_transactions[request.id] = request
        return ptac_pb2.Empty()

    def handlePBOrderbook(self, request: ptac_pb2.PBOrderbook, context):
        _env.wholesale_store.orderbooks[request.id] = request
        return ptac_pb2.Empty()

    def handlePBDistributionTransaction(self, request, context):
        warn_about_grpc_not_implemented()
        return ptac_pb2.Empty()

    def handlePBCapacityTransaction(self, request, context):
        warn_about_grpc_not_implemented()
        return ptac_pb2.Empty()

    def handlePBMarketBootstrapData(self, request, context):
        warn_about_grpc_not_implemented()
        return ptac_pb2.Empty()

    def handlePBWeatherForecast(self, request: ptac_pb2.PBWeatherForecast, context):
        _env.weather_store.weather_forecasts[request.currentTimeslot] = request
        for pred in request.predictions:
            _env.weather_store.weather_predictions[(request.currentTimeslot, pred.forecastTime)] = pred
        return ptac_pb2.Empty()

    def handlePBWeatherReport(self, request: ptac_pb2.PBWeatherReport, context):
        _env.weather_store.weather_reports[request.currentTimeslot] = request
        return ptac_pb2.Empty()

    def handlePBBalanceReport(self, request, context):
        warn_about_grpc_not_implemented()
        return ptac_pb2.Empty()


class PortfolioManagerService(ptac_grpc.PortfolioManagerServiceServicer):
    def handlePBCustomerBootstrapData(self, request, context):
        warn_about_grpc_not_implemented()
        return ptac_pb2.Empty()

    def handlePBTariffSpecification(self, request, context):
        warn_about_grpc_not_implemented()
        return ptac_pb2.Empty()

    def handlePBTariffStatus(self, request, context):
        warn_about_grpc_not_implemented()
        return ptac_pb2.Empty()

    def handlePBTariffTransaction(self, request, context):
        warn_about_grpc_not_implemented()
        return ptac_pb2.Empty()

    def handlePBTariffRevoke(self, request, context):
        warn_about_grpc_not_implemented()
        return ptac_pb2.Empty()

    def handlePBBalancingControlEvent(self, request, context):
        warn_about_grpc_not_implemented()
        return ptac_pb2.Empty()

    
class ConnectionService(ptac_grpc.ConnectionServiceServicer):
    def pingpong(self, request, context):
        log.info("ping received")
        return ptac_pb2.Empty()


# Helper methods
################################################################################
def warn_about_grpc_not_implemented():
        log.info(GRPC_METHOD_NOT_IMPLEMENTED)
        traceback.print_stack()


class CallLogInterceptor(grpc.ServerInterceptor):

    def __init__(self):
        super().__init__()
        log.info("logging interceptor loaded")

    def intercept_service(self, continuation, handler_call_details):
        #log.info("call received")
        return continuation(handler_call_details)
