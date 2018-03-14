# Starting the GRPC listeners
from queue import Queue
from threading import Thread

import util.id_generator as idg
import grpc
import tacgrpc.grpc_pb2_grpc as tac
import tacgrpc.grpc_pb2 as model

_channel = grpc.insecure_channel('localhost:1234')
_message_stub = tac.ServerMessagesStreamStub(_channel)

_out_counter = 0
_out_queue = Queue()
_in_queue = Queue()


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


def _connect_incoming():
    # handle incoming messages
    for msg in _message_stub.registerListener(model.Booly(value=True)):
        _in_queue.put(msg.rawMessage)


def _connect_outgoing():
    # register the iterator with the grpc stub
    _message_stub.registerEventSource(iter(_out_queue.get, None))
