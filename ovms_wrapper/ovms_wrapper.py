# OpenVINO Model Server wrapper API

import logging, sys
import grpc
import cv2
import numpy as np

# Suppress Tensorflow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow import make_tensor_proto
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc, model_service_pb2_grpc
from tensorflow_serving.apis import get_model_metadata_pb2, get_model_status_pb2

class OpenVINO_Model_Server:
    state_names = { 0: "UNKNOWN", 10: "START", 20: "LOADING", 30: "AVAILABLE", 40: "UNLOADING", 50: "END" }
    dtype_names = ['DT_INVALID', 'DT_FLOAT', 'DT_DOUBLE', 'DT_INT32', 'DT_UINT8', 'DT_INT16','DT_INT8', 
                   'DT_STRING', 'DT_COMPLEX64', 'DT_INT64', 'DT_BOOL', 'DT_QINT8', 'DT_QUINT8', 'DT_QINT32',
                   'DT_BFLOAT16', 'DT_QINT16', 'DT_QUINT16', 'DT_UINT16', 'DT_COMPLEX128', 'DT_HALF', 
                   'DT_RESOURCE', 'DT_VARIANT', 'DT_UINT32', 'DT_UINT64']

    class OVMS_model_info:
        def __init__(self):
            self.logger = logging.getLogger('OVMS_model')
            sh = logging.StreamHandler(sys.stdout)
            sh.setLevel(logging.INFO)
            fmt = logging.Formatter("[%(levelname)s] %(name)s : %(message)s")
            sh.setFormatter(fmt)
            self.logger.addHandler(sh)

            self.name = None
            self.inputs = []
            self.outputs = []
            self.result = None
            self.ovms = None
            self.available = False

        # inputs: Dictionary for input blobs
        # 'blobname1':contents, 'blobname2':contents, ...})
        def raw_infer(self, inputs, timeout:float=10.0):
            if self.available == False:
                self.logger.error('Model "{}" is not available'.format(self.name))
                return None
            # create a request for inference
            request = predict_pb2.PredictRequest()
            request.model_spec.name = self.model_name
            for idx, (bname, bval) in enumerate(inputs.items()):
                request.inputs[bname].CopyFrom(make_tensor_proto(bval, shape=bval.shape))
            # submit infer request
            self.result = self.ovms.stub.Predict(request, timeout) 
            return self.result

        # Parse and translate inference result to a dictionary style data
        # { 'outblob1':result, 'outblob2':result, ...}
        def parse_results(self):
            if self.available == False:
                self.logger.error('Model "{}" is not available'.format(self.name))
                return None
            result = {}
            for outblob in self.outputs:
                tensor = self.result.outputs[outblob['name']]
                shape = [dim.size for dim in tensor.tensor_shape.dim]
                content = np.frombuffer(tensor.tensor_content, dtype=np.float32).reshape(shape)
                result[outblob['name']] = content
            return result

        def single_image_infer(self, img, timeout:float=10.0):
            if self.available == False:
                self.logger.error('Model "{}" is not available'.format(self.name))
                return None
            inblob = self.inputs[0]
            img = cv2.resize(img, inblob['shape'][2:])
            img = img.transpose((2,0,1))
            img = img.reshape(inblob['shape']).astype(np.float32) # input is FP32
            self.raw_infer({inblob['name']:img})
            return self.parse_results()

    def __init__(self):
        self.logger = logging.getLogger('OVMS_wrapper')
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(levelname)s] %(name)s : %(message)s")
        sh.setFormatter(fmt)
        self.logger.addHandler(sh)

    def connect(self, grpc_ip_addr:str, grpc_port:int):
        self.ipaddr = grpc_ip_addr
        self.port = grpc_port
        self.channel = grpc.insecure_channel("{}:{}".format(grpc_ip_addr, grpc_port))
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)

    def get_model_status(self, model_name:str, model_version:int=1, timeout:float=10.0):
        stub = model_service_pb2_grpc.ModelServiceStub(self.channel)
        request = get_model_status_pb2.GetModelStatusRequest()
        request.model_spec.name = model_name
        request.model_spec.version.value = model_version
        try:
            result = stub.GetModelStatus(request, timeout)
        except grpc._channel._InactiveRpcError:
            self.logger.error('gRPC server connection error {}:{}'.format(self.ipaddr, self.port))
            return []
        status = []
        for i in result.model_version_status:
            status.append({ 'version':i.version, 'state':OpenVINO_Model_Server.state_names[i.state], 'error_code':i.status.error_code, 'error_msg':i.status.error_message})
        return status

    def open_model(self, model_name:str, timeout:float=10.0):
        model = self.OVMS_model_info()
        model.model_name = model_name
        request = get_model_metadata_pb2.GetModelMetadataRequest()
        request.model_spec.name = model_name
        request.metadata_field.append('signature_def')
        model.inputs = []
        model.outputs = []
        model.available = False
        try:
            # Obtain model information from OVMS
            result = self.stub.GetModelMetadata(request, timeout)
        except grpc._channel._InactiveRpcError:
            self.logger.error('gRPC server connection error {}:{}'.format(self.ipaddr, self.port))
            return model
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
            dtype = OpenVINO_Model_Server.dtype_names[self.serving_inputs[key].dtype]
            model.inputs.append({'name':name, 'shape':shape, 'dtype':dtype})
        # parse output blob info
        for key in self.serving_outputs.keys():
            name = self.serving_outputs[key].name
            shape = [dim.size for dim in self.serving_outputs[key].tensor_shape.dim]
            dtype = OpenVINO_Model_Server.dtype_names[self.serving_outputs[key].dtype]
            model.outputs.append({'name':name, 'shape':shape, 'dtype':dtype})
        self.logger.info('input/output blob info: {} / {}'.format(model.inputs, model.outputs))
        model.ovms = self
        model.available = True
        return model
