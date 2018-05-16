from pydispatch import dispatcher

from agent_components.demand import data
from communication.grpc_messages_pb2 import *
from communication.pubsub.signals import *
from agent_components.demand.learning import learner


def handle_timeslot_complete(sender, msg: PBTimeslotComplete):
    data.calculate_current_timestep(msg)
    training_data = data.calculate_training_data(msg)



def handle_tariff_transaction(sender, msg: PBTariffTransaction):
    data.update(msg)


def handle_competition(sender, msg: PBCompetition):
    pass

def handle_bootstrap_data(sender, msg: PBCustomerBootstrapData):
    """
    This is a message that holds a number of usages for a customer. We take it for truth and put it into our usage sums
    """
    data.update_with_bootstrap(msg)

# registering listeners
dispatcher.connect(handle_timeslot_complete, signal=PB_TIMESLOT_COMPLETE, sender=dispatcher.Any)
dispatcher.connect(handle_tariff_transaction, signal=PB_TARIFF_TRANSACTION, sender=dispatcher.Any)
dispatcher.connect(handle_competition, signal=PB_COMPETITION, sender=dispatcher.Any)
