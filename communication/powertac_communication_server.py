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
from queue import Queue

import grpc
from grpc import _server

import communication.grpc_messages_pb2 as ptac_pb2
import communication.grpc_messages_pb2_grpc as ptac_grpc
import util.config as cfg
from communication.pubsub.grpc_adapter import publish_pb_message
from util import id_generator
from util.strings import GRPC_SERVER_STARTING

log = logging.getLogger(__name__)
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

# Server Bootstrap
################################################################################


def serve():
    """Starts the grpc listener on the local machine"""
    server = grpc.server(executor, interceptors=[CallLogInterceptor()])
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


class SubmitService(ptac_grpc.SubmitServiceServicer):
    def __init__(self):
        self._order_queue = asyncio.Queue()
        self._tariff_spec_queue = asyncio.Queue()
        self._tariff_revoke_queue = asyncio.Queue()

    def send_order(self, msg: ptac_pb2.PBOrder):
        self._order_queue.put_nowait(msg)

    def send_tariff_revoke(self, msg: ptac_pb2.PBOrder):
        self._order_queue.put_nowait(msg)

    def send_tariff_spec(self, msg: ptac_pb2.PBTariffSpecification):
        self._tariff_spec_queue.put_nowait(msg)

    async def submitOrder(self, request, context):
        """DO NOT CALL from python. This is the API to the adapter"""
        while True:
            it = await self._order_queue.get()
            yield it

    async def submitTariffRevoke(self, request, context):
        """DO NOT CALL from python. This is the API to the adapter"""
        while True:
            it = await self._tariff_revoke_queue.get()
            yield it

    async def submitTariffSpec(self, request, context):
        """DO NOT CALL from python. This is the API to the adapter"""
        while True:
            it = await self._tariff_spec_queue.get()
            yield it


# global submit instance
submit_service = SubmitService()


# Handler Methods. Three classes for the corresponding Java Classes
################################################################################

class ContextManagerService(ptac_grpc.ContextManagerServiceServicer):
    async def handlePBBankTransaction(self, request, context):
        """in java, these are overloaded with different class types.
        GRPC doesn't allow same names so we name them as so
        handle<messagetype>()
        """

        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBCashPosition(self, request, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBDistributionReport(self, request, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBCompetition(self, request, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBProperties(self, request, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()


class MarketManagerService(ptac_grpc.MarketManagerServiceServicer):
    async def handlePBActivate(self, request, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBCompetition(self, request, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBBalancingTransaction(self, request, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBClearedTrade(self, request: ptac_pb2.PBClearedTrade, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBMarketPosition(self, request: ptac_pb2.PBMarketPosition, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBMarketTransaction(self, request: ptac_pb2.PBMarketTransaction, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBOrderbook(self, request: ptac_pb2.PBOrderbook, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBDistributionTransaction(self, request, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBCapacityTransaction(self, request, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBMarketBootstrapData(self, request, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBWeatherForecast(self, request: ptac_pb2.PBWeatherForecast, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBWeatherReport(self, request: ptac_pb2.PBWeatherReport, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBBalanceReport(self, request, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()


class PortfolioManagerService(ptac_grpc.PortfolioManagerServiceServicer):
    async def handlePBCustomerBootstrapData(self, request, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBTariffSpecification(self, request, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBTariffStatus(self, request, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBTariffTransaction(self, request, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBTariffRevoke(self, request, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBBalancingControlEvent(self, request, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()


class ConnectionService(ptac_grpc.ConnectionServiceServicer):
    async def pingpong(self, request, context):
        log.info("ping received")
        return ptac_pb2.Empty()


class GameService(ptac_grpc.GameServiceServicer):
    async def handlePBSimPause(self, request, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBTimeslotComplete(self, request, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBSimResume(self, request, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBTimeslotUpdate(self, request, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBSimEnd(self, request, context):
        publish_pb_message(request)
        return ptac_pb2.Empty()

    async def handlePBBrokerAccept(self, request: ptac_pb2.PBBrokerAccept, context):
        #ignoring the key, we only use the prefix because the key is added in java to the xml string
        id_generator.set_prefix(request.prefix)
        return ptac_pb2.Empty()


class ExtraSpyMessageManagerService(ptac_grpc.ExtraSpyMessageManagerServiceServicer):
    async def handlePBOrder(self, request, context):
        # log.info('received a spied upon order message')
        return ptac_pb2.Empty()


# Helper methods
################################################################################

class CallLogInterceptor(grpc.ServerInterceptor):

    def __init__(self):
        super().__init__()
        log.info("logging interceptor loaded")

    def intercept_service(self, continuation, handler_call_details):
        # log.info("call received")
        return continuation(handler_call_details)


# Monkey patching along the lines of https://gist.github.com/seglberg/0b4487b57b4fd425c56ad72aba9971be
# discussed here: https://github.com/grpc/grpc/issues/6046
################################################################################

def _loop_mgr(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

    # If we reach here, the loop was stopped.
    # We should gather any remaining tasks and finish them.
    pending = asyncio.Task.all_tasks(loop=loop)
    if pending:
        loop.run_until_complete(asyncio.gather(*pending))


class AsyncioExecutor(futures.Executor):

    def __init__(self, *, loop=None):

        super().__init__()
        self._shutdown = False
        self._loop = loop or asyncio.get_event_loop()
        self._thread = threading.Thread(target=_loop_mgr, args=(self._loop,),
                                        daemon=True)
        self._thread.start()

    def submit(self, fn, *args, **kwargs):

        if self._shutdown:
            raise RuntimeError('Cannot schedule new futures after shutdown')

        if not self._loop.is_running():
            raise RuntimeError("Loop must be started before any function can "
                               "be submitted")

        if inspect.iscoroutinefunction(fn):
            coro = fn(*args, **kwargs)
            return asyncio.run_coroutine_threadsafe(coro, self._loop)

        else:
            func = functools.partial(fn, *args, **kwargs)
            return self._loop.run_in_executor(None, func)

    def shutdown(self, wait=True):
        self._loop.stop()
        self._shutdown = True
        if wait:
            self._thread.join()


# --------------------------------------------------------------------------- #


async def _call_behavior(rpc_event, state, behavior, argument, request_deserializer):
    context = _server._Context(rpc_event, state, request_deserializer)
    try:
        return await behavior(argument, context), True
    except Exception as e:  # pylint: disable=broad-except
        with state.condition:
            if e not in state.rpc_errors:
                details = 'Exception calling application: {}'.format(e)
                _server.logging.exception(details)
                _server._abort(state, rpc_event.operation_call,
                               _server.cygrpc.StatusCode.unknown, _server._common.encode(details))
        return None, False


async def _take_response_from_response_iterator(rpc_event, state, response_iterator):
    try:
        return await response_iterator.__anext__(), True
    except StopAsyncIteration:
        return None, True
    except Exception as e:  # pylint: disable=broad-except
        with state.condition:
            if e not in state.rpc_errors:
                details = 'Exception iterating responses: {}'.format(e)
                _server.logging.exception(details)
                _server._abort(state, rpc_event.operation_call,
                               _server.cygrpc.StatusCode.unknown, _server._common.encode(details))
        return None, False


async def _unary_response_in_pool(rpc_event, state, behavior, argument_thunk,
                                  request_deserializer, response_serializer):
    argument = argument_thunk()
    if argument is not None:
        response, proceed = await _call_behavior(rpc_event, state, behavior,
                                                 argument, request_deserializer)
        if proceed:
            serialized_response = _server._serialize_response(
                rpc_event, state, response, response_serializer)
            if serialized_response is not None:
                _server._status(rpc_event, state, serialized_response)


async def _stream_response_in_pool(rpc_event, state, behavior, argument_thunk,
                                   request_deserializer, response_serializer):
    argument = argument_thunk()
    if argument is not None:
        # Notice this calls the normal `_call_behavior` not the awaitable version.
        response_iterator, proceed = _server._call_behavior(
            rpc_event, state, behavior, argument, request_deserializer)
        if proceed:
            while True:
                response, proceed = await _take_response_from_response_iterator(
                    rpc_event, state, response_iterator)
                if proceed:
                    if response is None:
                        _server._status(rpc_event, state, None)
                        break
                    else:
                        serialized_response = _server._serialize_response(
                            rpc_event, state, response, response_serializer)
                        print(response)
                        if serialized_response is not None:
                            print("Serialized Correctly")
                            proceed = _server._send_response(rpc_event, state,
                                                             serialized_response)
                            if not proceed:
                                break
                        else:
                            break
                else:
                    break


#monkey patching here
_server._unary_response_in_pool = _unary_response_in_pool
_server._stream_response_in_pool = _stream_response_in_pool
executor = AsyncioExecutor()
