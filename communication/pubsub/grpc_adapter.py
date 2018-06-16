"""
this module allows easy sharing of arbitrary grpc messages via the event pubsub architecture.
It works async, meaning the client takes the messages of the Java adapter and immidiately returns an OK instead of
blocking the connection until whatever handlers are completed.
"""
import asyncio
import logging

from pydispatch import dispatcher


log = logging.getLogger(__name__)

def publish_pb_message(pb_message, loop=None):
    """
    Publishes a message from grpc based on its name via the pubsub architecture.
    :param pb_message: the message to share with the other components
    :return:
    """
    if loop is None:
        loop = asyncio.get_event_loop()

    #asyncio.ensure_future(send_message_async(pb_message), loop=loop)
    send_message_async(pb_message)

def send_message_async(pb_message):
    signal = pb_message.DESCRIPTOR.name
    log.info("dispatching {}".format(signal))
    dispatcher.send(signal=signal, msg=pb_message, sender=dispatcher.Anonymous)

#async def send_message_async(pb_message):
#    signal = pb_message.DESCRIPTOR.name
#    log.info("dispatching {}".format(signal))
#    dispatcher.send(signal=signal, msg=pb_message, sender=dispatcher.Anonymous)
