@echo off

: Create a temporaty venv to install OpenVINO and download several models using OpenVINO model downloader tool
python -m venv venv-ov-temp__
call venv-ov-temp__\Scripts\activate
python -m pip install openvino-dev tensorflow

: Download Resnet-50, Googlenet-v1, and face-detection-0200 models
: face-detection-0200 is in IR model format. No need for conversion.
omz_downloader --name resnet-50-tf,googlenet-v1-tf,face-detection-0200 --precisions FP16
omz_converter  --name resnet-50-tf,googlenet-v1-tf                     --precisions FP16

: Deactivate venv and remove temporary environment
call deactivate
rd /s /q venv-ov-temp__

: Prepare directries for the model repository
mkdir .\ovms_model_repository\models\resnet-50-tf\1
mkdir .\ovms_model_repository\models\googlenet-v1-tf\1
mkdir .\ovms_model_repository\models\face-detection-0200\1
: Copy required files to the model repository (mapping_config.json is an optional)
copy .\public\resnet-50-tf\FP16\*       .\ovms_model_repository\models\resnet-50-tf\1\
copy .\public\googlenet-v1-tf\FP16\*    .\ovms_model_repository\models\googlenet-v1-tf\1\
copy .\intel\face-detection-0200\FP16\* .\ovms_model_repository\models\face-detection-0200\1\
copy .\model-config.json                .\ovms_model_repository\models\config.json
copy .\mapping_config-resnet-50-tf.json .\ovms_model_repository\models\resnet-50-tf\1\mapping_config.json

: ./ovms_model_repository/
:  + models
:  | + resnet-50-tf
:  | | + 1                     (<- model version number)
:  | | | + resnet-50-tf.xml
:  | | | + resnet-50-tf.bin
:  | | | + mapping_config.json ( <- optional)
:  | + googlenet-v1-tf
:  | | + 1
:  | | | + googlenet-v1-tf.xml
:  | | | + googlenet-v1-tf.bin
:  | + face-detection-0200
:  | | + 1
:  | | | + face-detection-0200.xml
:  | | | + face-detection-0200.bin
:  | + config.json

: ## How to start OVMS on Windows
: 
: docker run -d --rm --name ovms ^
:  -v c:/<abs_path>/ovms_model_repository/models:/opt/models ^
:  -p 9000:9000 openvino/model_server:latest ^
:  --config_path=/opt/models/config.json ^
:  --port 9000
