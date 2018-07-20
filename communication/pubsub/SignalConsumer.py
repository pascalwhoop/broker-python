
class SignalConsumer:
    """An Interface that Signal Consumers should implement"""
    def __init__(self):
        pass

    def subscribe(self):
        """Implement this to hook all signal listeners up to the component"""
        raise NotImplementedError

    def unsubscribe(self):
        """Implement this to have the component removed from the pubsub infrastructure.
        unsubscribe manually in implementation"""
        raise NotImplementedError
