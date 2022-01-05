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
bdata = model.image_preprocess(inblob, img)
# model.image_preprocess() equivalent
#  bdata = cv2.resize(img, inblob['shape'][2:])
#  bdata = bdata.transpose((2,0,1))
#  bdata = ibdata.reshape(inblob['shape']).astype(inblob['dtype_npy'])

model.raw_infer({ inblob['name']:bdata })       # Raw Infer
res = model.parse_results()                     # Parse (decode) inference result
result = res[model.outputs[0]['name']]

# display result
nu = np.array(result)
ma = np.argmax(nu)
print("Result:", ma)