import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import grpc
import sys

import communication.grpc_messages_pb2 as ptac_pb2
import communication.grpc_messages_pb2_grpc as ptac_grpc
import util.config as cfg
from util.strings import GRPC_METHOD_NOT_IMPLEMENTED, GRPC_SERVER_STARTING

log = logging.getLogger(__name__)
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


# Server Bootstrap
################################################################################

def serve():
    """Starts the grpc listener on the local machine"""
    server = grpc.server(ThreadPoolExecutor(max_workers=1), interceptors=[CallLogInterceptor()])
    ptac_grpc.add_ContextManagerServiceServicer_to_server(ContextManagerService(),     server)
    ptac_grpc.add_MarketManagerServiceServicer_to_server(MarketManagerService(),       server)
    ptac_grpc.add_PortfolioManagerServiceServicer_to_server(PortfolioManagerService(), server)
    ptac_grpc.add_ConnectionServiceServicer_to_server(ConnectionService(), server)


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

    def handlePBClearedTrade(self, request, context):
        warn_about_grpc_not_implemented()
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

    def handlePBMarketPosition(self, request, context):
        warn_about_grpc_not_implemented()
        return ptac_pb2.Empty()

    def handlePBMarketTransaction(self, request, context):
        warn_about_grpc_not_implemented()
        return ptac_pb2.Empty()

    def handlePBOrderbook(self, request, context):
        warn_about_grpc_not_implemented()
        return ptac_pb2.Empty()

    def handlePBWeatherForecast(self, request, context):
        warn_about_grpc_not_implemented()
        return ptac_pb2.Empty()

    def handlePBWeatherReport(self, request, context):
        warn_about_grpc_not_implemented()
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
        log.warning(GRPC_METHOD_NOT_IMPLEMENTED)
        traceback.print_stack()


class CallLogInterceptor(grpc.ServerInterceptor):

    def __init__(self):
        super().__init__()
        log.info("logging interceptor loaded")

    def intercept_service(self, continuation, handler_call_details):
        #log.info("call received")
        return continuation(handler_call_details)
