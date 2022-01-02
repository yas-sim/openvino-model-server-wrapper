from ovms_wrapper import OpenVINO_Model_Server

import cv2
import numpy as np

ovms = OpenVINO_Model_Server()
ovms.connect('127.0.0.1', 9000)
model = ovms.open_model('resnet_50')

image_file  = 'ovms/lib/python3.8/site-packages/skimage/data/rocket.jpg'

img = cv2.imread(image_file)                # Read an image
res = ovms.single_image_infer(model, img)   # Infer
result = res[model.outputs[0]['name']]

# display result
nu = np.array(result)
ma = np.argmax(nu)
print("Result:", ma)
