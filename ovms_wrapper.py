# OpenVINO Model Server wrapper API

import logging, sys
import grpc
import cv2
import numpy as np

# Suppress Tensorflow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import get_model_metadata_pb2

class OpenVINO_Model_Server:
    class OVMS_model_info:
        def __init__(self):
            self.name = None
            self.inputs = []
            self.outputs = []
            self.result = None

    def __init__(self):
        self.logger = logging.getLogger('OVMS_wrapper')
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        #fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%dT%H:%M:%S")
        fmt = logging.Formatter("[%(levelname)s] %(name)s : %(message)s")
        sh.setFormatter(fmt)
        self.logger.addHandler(sh)

    def connect(self, grpc_ip_addr:str, grpc_port:int):
        self.ipaddr = grpc_ip_addr
        self.port = grpc_port
        self.channel = grpc.insecure_channel("{}:{}".format(grpc_ip_addr, grpc_port))
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)

    def open_model(self, model_name:str, timeout:float=10.0):
        model = self.OVMS_model_info()
        model.model_name = model_name
        request = get_model_metadata_pb2.GetModelMetadataRequest()
        request.model_spec.name = model_name
        request.metadata_field.append('signature_def')
        model.inputs = []
        model.outputs = []
        try:
            # Obtain model information from OVMS
            result = self.stub.GetModelMetadata(request, timeout)
        except grpc._channel._InactiveRpcError:
            self.logger.error('gRPC server connection error {}:{}'.format(self.ipaddr, self.port))
            return
        signature_def = result.metadata['signature_def']
        signature_map = get_model_metadata_pb2.SignatureDefMap()
        signature_map.ParseFromString(signature_def.value)
        serving_default = signature_map.ListFields()[0][1]['serving_default']
        self.serving_inputs = serving_default.inputs
        self.serving_outputs = serving_default.outputs
        # parse input blob info
        for key in self.serving_inputs.keys():
            name = self.serving_inputs[key].name
            shape = [dim.size for dim in self.serving_inputs[key].tensor_shape.dim]
            dtype = self.serving_inputs[key].dtype
            model.inputs.append({'name':name, 'shape':shape, 'dtype':dtype})
        # parse output blob info
        for key in self.serving_outputs.keys():
            name = self.serving_outputs[key].name
            shape = [dim.size for dim in self.serving_outputs[key].tensor_shape.dim]
            dtype = self.serving_outputs[key].dtype
            model.outputs.append({'name':name, 'shape':shape, 'dtype':dtype})
        self.logger.info('input/output blob info: {} / {}'.format(model.inputs, model.outputs))
        model.ovms = self
        return model

    # inputs: Dictionary for input blobs
    # 'blobname1':contents, 'blobname2':contents, ...})
    def raw_infer(self, model, inputs, timeout:float=10.0):
        # create a request for inference
        request = predict_pb2.PredictRequest()
        request.model_spec.name = model.model_name
        for idx, (bname, bval) in enumerate(inputs.items()):
            request.inputs[bname].CopyFrom(tf.make_tensor_proto(bval, shape=bval.shape))
        # submit infer request
        model.result = self.stub.Predict(request, timeout) 
        return model.result

    # Parse and translate inference result to a dictionary style data
    # { 'outblob1':result, 'outblob2':result, ...}
    def parse_results(self, model):
        result = {}
        for outblob in model.outputs:
            result[outblob['name']] = tf.make_ndarray(model.result.outputs[outblob['name']])
        return result

    def single_image_infer(self, model, img, timeout:float=10.0):
        inblob = model.inputs[0]
        img = cv2.resize(img, inblob['shape'][2:])
        img = img.transpose((2,0,1))
        img = img.reshape(inblob['shape']).astype(np.float32) # input is FP32
        self.raw_infer(model, {inblob['name']:img})
        return self.parse_results(model)
