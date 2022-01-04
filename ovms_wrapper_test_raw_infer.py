from ovms_wrapper.ovms_wrapper import OpenVINO_Model_Server

import cv2
import numpy as np

ovms = OpenVINO_Model_Server()
ovms.connect('127.0.0.1', 9000)
#print(ovms.get_model_status('resnet_50'))
model = ovms.open_model('resnet_50')
print(model.inputs, model.outputs)
inblob = model.inputs[0]

# curl -O https://raw.githubusercontent.com/intel-iot-devkit/smart-video-workshop/master/Labs/daisy.jpg
image_file  = 'daisy.jpg'
img = cv2.imread(image_file)                    # Read an image
img = cv2.resize(img, inblob['shape'][2:])
img = img.transpose((2,0,1))
img = img.reshape(inblob['shape']).astype(inblob['dtype_npy'])
inblob_name = inblob['name']
model.raw_infer({ inblob_name:img })         # Raw Infer
res = model.parse_results()
result = res[model.outputs[0]['name']]

# display result
nu = np.array(result)
ma = np.argmax(nu)
print("Result:", ma)