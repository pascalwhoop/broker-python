# Starting the GRPC listeners
from queue import Queue
from threading import Thread

import util.id_generator as idg
import grpc

import tacgrpc.grpc_pb2 as model
import tacgrpc.grpc_pb2_grpc as tac

_channel = grpc.insecure_channel('localhost:1234')
_message_stub = tac.ServerMessagesStreamStub(_channel)

_out_counter = 0
_out_queue = Queue()
_in_queue = Queue()

# more interceptors can be added if so desired. They need to be able to handle xml as string
interceptors = [idg.broker_accept_intercept]


def reset():
    global _out_queue, _in_queue, _out_counter
    _out_counter = 0
    _out_queue = Queue()
    _in_queue = Queue()


def put(msg: str):
    global _out_counter
    _out_counter += 1
    x_msg = model.XmlMessage(counter=_out_counter, rawMessage=idg.key + msg)
    _out_queue.put(x_msg)


def get():
    return _in_queue.get()


def connect():
    """create 2 threads that connect to the server and read/write their messages from the blocking queues."""
    in_thread = Thread(target=_connect_incoming)
    out_thread = Thread(target=_connect_outgoing)
    in_thread.start()
    out_thread.start()
    return in_thread, out_thread


def intercept_maybe(msg):
    for i in interceptors:
        i(msg)


def _connect_incoming():
    # handle incoming messages
    for msg in _message_stub.registerListener(model.Booly(value=True)):
        intercept_maybe(msg.rawMessage)
        _in_queue.put(msg.rawMessage)


def _connect_outgoing():
    # register the iterator with the grpc stub
    _message_stub.registerEventSource(iter(_out_queue.get, None))
