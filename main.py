# Starting the GRPC listeners
import grpc
import tacgrpc.grpc_pb2_grpc as tac
import tacgrpc.grpc_pb2 as model

channel = grpc.insecure_channel('localhost:1234')
message_stub = tac.ServerMessagesStreamStub(channel)

# for props in context_stub.handlePBProperties(model.PBRequestStream(msg="Properties")):
#    print(props)

for msg in message_stub.registerListener(model.XmlMessage()):
    print(msg)
