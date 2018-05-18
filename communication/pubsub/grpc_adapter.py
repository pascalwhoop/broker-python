"""
this module allows easy sharing of arbitrary grpc messages via the event pubsub architecture.
"""
from pydispatch import dispatcher


def publish_pb_message(pb_message):
    """
    Publishes a message from grpc based on its name via the pubsub architecture.
    :param pb_message: the message to share with the other components
    :return:
    """
    signal = pb_message.DESCRIPTOR.name
    dispatcher.send(signal=signal, msg=pb_message, sender=dispatcher.Anonymous)
