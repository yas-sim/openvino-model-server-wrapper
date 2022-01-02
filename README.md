# OpenVINO Model Server wrapper API for Python
## Description
This project provides a Python wrapper class for OpenVINO Model Server.  
User can submit DL inference request to OVMS with just a few lines of code.  

## Sample code
```sh
from ovms_wrapper.ovms_wrapper import OpenVINO_Model_Server

import cv2
import numpy as np

ovms = OpenVINO_Model_Server()
ovms.connect('127.0.0.1', 9000)
status = ovms.get_model_status('resnet_50')
print(status)
model = ovms.open_model('resnet_50')

image_file  = 'ovms/lib/python3.8/site-packages/skimage/data/rocket.jpg'

print(model.inputs, model.outputs)

img = cv2.imread(image_file)                # Read an image
res = model.single_image_infer(img)         # Infer
result = res[model.outputs[0]['name']]

# display result
nu = np.array(result)
ma = np.argmax(nu)
print("Result:", ma)
```

## How to setup OpenVINO Model Server (Ubuntu)
Note: OVMS can run on Windows too. Please refer to the [official OVMS document](https://docs.openvino.ai/latest/openvino_docs_ovms.html) for details.  
```sh
sudo apt update && sudo apt install -y python3-venv
python -m pip install tensorflow tensorflow-serving-api
```
- Create Python virtual env, install OpenVINO, and prepare an IR model  
Installing OpenVINO just for downloading a DL model and converting it into OpenVINO IR model. This is not required if you already have the IR models.  
```sh
python3 -m venv ovms
. ovms/bin/activate
python -m pip install openvino-dev
omz_downloader --name resnet-50-tf
omz_converter --name resnet-50-tf --precisions FP16
deactivate
```
- Start OpenVINO Model Server as Docker container
```sh
docker run -d --rm -v $PWD/public/resnet-50-tf/FP16:/models/resnet50/1 -p 9000:9000 openvino/model_server:latest --model_path /models/resnet50 --model_name resnet_50 --port 9000
```
OVMS will start serving the Resnet-50 model as model-name='resnet_50', model-version=1, and gRPC-port=9000.

