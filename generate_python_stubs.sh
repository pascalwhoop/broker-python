#!/usr/bin/env bash
#builds the entities using protobuf
protoc --plugin=protoc-gen-mypy=venv/bin/protoc-gen-mypy \
    --python_out=./communication \
    --mypy_out=./communication \
    --proto_path=../grpc-adapter/adapter/src/main/proto/ ../grpc-adapter/adapter/src/main/proto/grpc_messages.proto
# builds the services
python -m grpc_tools.protoc -I../grpc-adapter/adapter/src/main/proto/ --grpc_python_out=./communication ../grpc-adapter/adapter/src/main/proto/grpc_messages.proto
#fixing dependency
sed -i -e 's/import grpc_messages_pb2/import communication.grpc_messages_pb2/' communication/grpc_messages_pb2_grpc.py
