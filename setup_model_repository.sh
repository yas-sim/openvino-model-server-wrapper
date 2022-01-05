#!/usr/bin/env /usr/bin/bash

# Create a temporaty venv to install OpenVINO and download several models using OpenVINO model downloader tool
python3 -m venv venv-ov-temp__
. venv-ov-temp__/bin/activate
python3 -m pip install openvino-dev tensorflow

# Download Resnet-50, Googlenet-v1, and face-detection-0200 models
# face-detection-0200 is in IR model format. No need for conversion.
omz_downloader --name resnet-50-tf,googlenet-v1-tf,face-detection-0200 --precisions FP16
omz_converter  --name resnet-50-tf,googlenet-v1-tf                     --precisions FP16

# Deactivate venv and remove temporary environment
deactivate
rm -rf venv-ov-temp__

# Prepare directries for the model repository
mkdir -p ./ovms_model_repository/models/resnet-50-tf/1
mkdir -p ./ovms_model_repository/models/googlenet-v1-tf/1
mkdir -p ./ovms_model_repository/models/face-detection-0200/1
# Copy required files to the model repository (mapping_config.json is an optional)
cp ./public/resnet-50-tf/FP16/*       ./ovms_model_repository/models/resnet-50-tf/1/
cp ./public/googlenet-v1-tf/FP16/*    ./ovms_model_repository/models/googlenet-v1-tf/1/
cp ./intel/face-detection-0200/FP16/* ./ovms_model_repository/models/face-detection-0200/1/
cp ./model-config.json                ./ovms_model_repository/models/config.json
cp ./mapping_config-resnet-50-tf.json ./ovms_model_repository/models/resnet-50-tf/1/mapping_config.json

# ./ovms_model_repository/
#  + models
#  | + resnet-50-tf
#  | | + 1                     (<- model version number)
#  | | | + resnet-50-tf.xml
#  | | | + resnet-50-tf.bin
#  | | | + mapping_config.json ( <- optional)
#  | + googlenet-v1-tf
#  | | + 1
#  | | | + googlenet-v1-tf.xml
#  | | | + googlenet-v1-tf.bin
#  | + face-detection-0200
#  | | + 1
#  | | | + face-detection-0200.xml
#  | | | + face-detection-0200.bin
#  | + config.json
