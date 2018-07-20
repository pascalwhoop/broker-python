"""
this module allows easy sharing of arbitrary grpc messages via the event pubsub architecture.
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
    signal = pb_message.DESCRIPTOR.name
    log.info("dispatching {}".format(signal))
    try:
        dispatcher.send(signal=signal, msg=pb_message, sender=dispatcher.Anonymous)
    except Exception as e:
        # catching all exceptions that boil up here, to avoid the agent from not working anymore
        log.exception(e)

