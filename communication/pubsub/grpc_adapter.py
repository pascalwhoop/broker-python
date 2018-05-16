"""
this module allows easy sharing of arbitrary grpc messages via the event pubsub architecture.
"""
from pydispatch import dispatcher


def publish_grpc_message(grpc_message):
    """
    Publishes a message from grpc based on its name via the pubsub architecture.
    :param grpc_message: the message to share with the other components
    :return:
    """
    signal = grpc_message.DESCRIPTOR.name
    dispatcher.send(signal=signal, msg=grpc_message, sender=dispatcher.Anonymous)
