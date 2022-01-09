# How to Setup OpenVINO Model Server for the demo programs

1. Download required IR models
```sh
python3 -m venv venv__temp
. venv__temp/bin/activate
python3 -m pip install openvino-dev
omz_downloader --list models.lst --precisions FP16
deactivate
```
* (optional) Now you can remove the temporary virtualenv.
```sh
rm -rf venv__temp
```

2. Create OVMS model repository  
After you generate the model repository, the original model directory and files are no longer needed.  
```sh
python3 ../../model
-repo-generator/setup_ovms_model_repo.py -m intel -o repo
```

3. Start OpenVINO Model Server with the model repository.  
```sh
docker run -d --rm --name ovms \
  -v $PWD/repo/models:/opt/models \
  -p 9000:9000 openvino/model_server:latest \
  --config_path=/opt/models/config.json \
  --port 9000
```
