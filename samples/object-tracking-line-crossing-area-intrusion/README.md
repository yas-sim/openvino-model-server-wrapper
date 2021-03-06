# Object Tracking with Line Crossing and Area Intrusion Detection Demo for OVMS wrapper

## Description  
This demo runs 'person-detection-0200' and 'person-reidentification-0277' model with OpenVINO model server using OVMS wrapper library.  
The demo detects person in the input frame and track them.  
![object-track](./resources/object-track.gif)

## How to run
1. Copy OVMS wrapper library  
- Ubuntu
```sh
cp -r ../../ovms_wrapper .
```
- Windows 10/11
```sh
xcopy /E ..\..\ovms_wrapper\ ovms_wrapper\
```

* (Optional) Copy gRPC handler codes  
If you have installed TF and TF-serving-api, you can skip this operation.  
You can use gRPC handler codes instead of TF and TF-serving-api. This is useful when you want to run this demo on a small devices such as Raspberry Pi.  

- Ubuntu
```sh
cp -r ../../_tensorflow ./tensorflow
cp -r ../../_tensorflow_serving ./tensorflow_serving
```
- Windows 10/11
```sh
xcopy /E ..\..\_tensorflow\ tensorflow\
xcopy /E ..\..\_tensorflow_serving\ tensorflow_serving\
```

3. Install prerequisites for the demo
- Ubuntu
```sh
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip setuptools
python3 -m pip install numpy opencv-python scipy munkres grpcio_tools
```
- Windows 10/11
```sh
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip setuptools
python -m pip install numpy opencv-python scipy munkres grpcio_tools
```

4. Setup and start OVMS  
Please refer to [`How to setup and start OpenVINO Model Server fot the demos`](../ovms_setup_for_demos) page to start OVMS.  

5. Run the demo program
- Ubuntu
```sh
python3 object-detection-and-line-cross.py
```
- Windows 10/11
```sh
python object-detection-and-line-cross.py
```