# Human Pose Estimation Demo for OVMS wrapper  

## Description  
This demo runs 'human-pose-estimation-0001' model with OpenVINO model server using OVMS wrapper library.  

## How to run
1. Build pose_extractor Python module  
Build a Python module and copy the built library file to the demo directory.
```sh
cd pose_extractor_src
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ../..
cp pose_extractor_src/build/pose_extractor/pose_extractor.so .
```

2. Copy OVMS wrapper library  
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
python3 -m pip install grpcio grpcio_tools numpy opencv-python
```

4. Setup and start OVMS  
Please refer to [`How to setup and start OpenVINO Model Server fot the demos`](../ovms_setup_for_demos) page to start OVMS.  

5. Run the demo program
```sh
python3 human-pose-estimation-2d.py
```

![image](resources/human-pose-demo.png)
