#!/usr/bin/env /usr/bin/bash
tf=tensorflow/core/framework
sv=tensorflow_serving/apis

# python3 -m pip install grpcio-tools

pushd tensorflow
find . -name "*.py" | xargs rm
popd
pushd tensorflow_serving
find . -name "*.py" | xargs rm
popd

python3 -m grpc_tools.protoc -I. --python_out=. \
 tensorflow/core/framework/tensor.proto \
 tensorflow/core/framework/tensor_shape.proto \
 tensorflow/core/framework/resource_handle.proto \
 tensorflow/core/framework/types.proto \
 tensorflow/core/framework/full_type.proto \
 tensorflow/core/framework/graph.proto \
 tensorflow/core/framework/op_def.proto \
 tensorflow/core/framework/node_def.proto \
 tensorflow/core/framework/function.proto \
 tensorflow/core/framework/attr_value.proto \
 tensorflow/core/framework/versions.proto \
 tensorflow/core/framework/variable.proto \
 tensorflow/core/example/example.proto \
 tensorflow/core/example/feature.proto \
 tensorflow/core/protobuf/struct.proto \
 tensorflow/core/protobuf/trackable_object_graph.proto \
 tensorflow/core/protobuf/saved_object_graph.proto \
 tensorflow/core/protobuf/saver.proto \
 tensorflow/core/protobuf/error_codes.proto \
 tensorflow/core/protobuf/meta_graph.proto

python3 -m grpc_tools.protoc -I. --python_out=. \
 tensorflow_serving/apis/predict.proto \
 tensorflow_serving/apis/classification.proto \
 tensorflow_serving/apis/model.proto \
 tensorflow_serving/apis/input.proto \
 tensorflow_serving/apis/status.proto \
 tensorflow_serving/apis/inference.proto \
 tensorflow_serving/apis/regression.proto \
 tensorflow_serving/apis/model_management.proto \
 tensorflow_serving/apis/get_model_metadata.proto \
 tensorflow_serving/apis/get_model_status.proto \
 tensorflow_serving/config/file_system_storage_path_source.proto \
 tensorflow_serving/config/logging_config.proto \
 tensorflow_serving/config/log_collector_config.proto \
 tensorflow_serving/config/model_server_config.proto

python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. \
 tensorflow_serving/apis/prediction_service.proto \
 tensorflow_serving/apis/model_service.proto

#from tensorflow.core.framework import tensor_pb2, tensor_shape_pb2
#from tensorflow_serving.apis import predict_pb2
#from tensorflow_serving.apis import prediction_service_pb2_grpc, model_service_pb2_grpc
#from tensorflow_serving.apis import get_model_metadata_pb2, get_model_status_pb2
