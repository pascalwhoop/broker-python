# Starting the GRPC listeners
from queue import Queue, Empty
from threading import Thread

import grpc

import util.id_generator as idg
import communication.grpc_pb2 as model
import communication.grpc_pb2_grpc as tac

_channel           = grpc.insecure_channel('localhost:1234')
_message_stub      = tac.ServerMessagesStreamStub(_channel)
_out_counter       = 0
_out_queue         = Queue()
_in_queue          = Queue()
_com_threads       = ()
_connected_flag    = False
_should_disconnect = False

# more interceptors can be added if so desired. They need to be able to handle xml as string
interceptors = [idg.broker_accept_intercept]


def reset_queues():
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
    """the public connect function. Creates a thread that manages all connectivity and returns to caller quickly"""
    conn_thread = Thread(target=_connect)
    conn_thread.start()
    return conn_thread


def _connect():
    """create 2 threads that connect to the server and read/write their messages from the blocking queues."""
    global _connected_flag, _should_disconnect
    _connected_flag = True
    _should_disconnect = False

    while not _should_disconnect:
        in_thread    = Thread(target=_connect_incoming)
        out_thread   = Thread(target=_connect_outgoing)

        in_thread.start()
        out_thread.start()
        global _com_threads
        _com_threads = in_thread, out_thread
        # waiting for comm threads to quit
        in_thread.join()
        out_thread.join()

def disconnect():
    """sets the disconnect flag True so that the child threads quit whenever possible"""
    global should_disconnect
    should_disconnect = True



def call_interceptors(msg):
    for i in interceptors:
        i(msg)


def _connect_incoming():
    # handle incoming messages
    try:
        for msg in _message_stub.registerListener(model.Booly(value=True)):
            call_interceptors(msg.rawMessage)
            _in_queue.put(msg.rawMessage)
            if _should_disconnect or  not _connected_flag:
                break #breaking loop, exiting thread and letting handler reconnect
    except Exception as e:
        # need to reconnect
        print(e)
        print("trying to reconnect")


def _connect_outgoing():
    global _connected_flag
    while _connected_flag or not _should_disconnect:
        try:
            # register the iterator with the grpc stub
            _message_stub.registerEventSource(iter(_out_queue.get, 1))
        except Empty:
            # can be ignored, we will get again in a second
            pass
        except Exception as e:
            # now we are in trouble. Break and reconnect
            _connected_flag = False
            print(e)
            print("trying to reconnect")


def handle_disconnect():
    _connected_flag = False
    _reconnect()

def _reconnect():
    pass
