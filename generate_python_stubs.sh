#!/usr/bin/env bash
protoc --plugin=protoc-gen-mypy=venv/bin/protoc-gen-mypy --python_out=./communication --mypy_out=./communication --proto_path=../sample-broker/src/main/proto/ ../sample-broker/src/main/proto/grpc_messages.proto
