#!/usr/bin/env bash
cd ..
echo "Assumes sample-broker project is sibling to this project"
echo "Generates python files from grpc.proto file in sibling project"
python -m grpc_tools.protoc -I../sample-broker/src/main/proto/ --python_out=./communication --grpc_python_out=./communication ../sample-broker/src/main/proto/grpc_messages.proto
sed -i "s/import grpc_/import tacgrpc.grpc_/g" "communication/grpc_messages_pb2_grpc.py"
#python3 -m grpc_tools.protoc -I../broker-adapter/src/main/proto/ --python_out=./tacgrpc --grpc_python_out=./communication ../broker-adapter/src/main/proto/grpc.proto
sed -i "s/import grpc_/import tacgrpc.grpc_/g" "communication/grpc_pb2_grpc.py"