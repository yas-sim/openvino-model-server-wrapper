from ovms_wrapper.ovms_wrapper import OpenVINO_Model_Server

import cv2
import numpy as np

ovms = OpenVINO_Model_Server()
ovms.connect('127.0.0.1', 9000)
#print(ovms.get_model_status('resnet-50-tf'))
model = ovms.open_model('resnet-50-tf')
print(model.inputs, model.outputs)

# curl -O https://raw.githubusercontent.com/intel-iot-devkit/smart-video-workshop/master/Labs/daisy.jpg
image_file  = 'daisy.jpg'
img = cv2.imread(image_file)                # Read an image
res = model.single_image_infer(img)         # Single Image Infer
result = res[model.outputs[0]['name']]

# display result
nu = np.array(result)
ma = np.argmax(nu)
print("Result:", ma)