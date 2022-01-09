# Object Tracking with Line Crossing and Area Intrusion Detection Demo for OVMS wrapper

## Description  
This demo runs 'person-detection-0200' and 'person-reidentification-0277' model with OpenVINO model server using OVMS wrapper library.  
The demo detects person in the input frame and track them.  
![object-track](./resources/object-track.gif)

## How to run
1. Copy OVMS wrapper library  
```sh
cp -r ../../ovms_wrapper .
```

* (Optional) Copy gRPC handler codes  
You can use those gRPC handler codes if you don't want to (or cannot) install TensorFlow and tensorflow-serving api.  
```sh
cp -r ../../_tensorflow ./tensorflow
cp -r ../../_tensorflow_serving ./tensorflow_serving
```

3. Install prerequisites for the demo
```sh
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip setuptools
python3 -m pip install numpy opencv-python scipy munkres grpcio_tools
```

4. Setup and start OVMS  
Please refer to [`How to setup and start OpenVINO Model Server fot the demos`](../ovms_setup_for_demos) page to start OVMS.  

5. Run the demo program
```sh
python3 object-detection-and-line-cross.py
```