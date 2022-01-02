```sh
sudo apt update && sudo apt install -y python3-venv
python -m pip install tensorflow tensorflow-serving-api
```
- Create Python virtual env and install OpenVINO
```sh
python3 -m venv ovms
. ovms/bin/activate
python -m pip install openvino-dev
```
```sh
omz_downloader --name resnet-50-tf
omz_converter --name resnet-50-tf --precisions FP16
```

```sh
docker run -d --rm -v $PWD/public/resnet-50-tf/FP16:/models/resnet50/1 -p 9000:9000 openvino/model_server:latest --model_path /models/resnet50 --model_name resnet_50 --port 9000
```

Model name = 'resnet_50', model version = 1, port = 9000 (gRPC)

