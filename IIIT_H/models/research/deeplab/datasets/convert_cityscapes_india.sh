#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Script to preprocess the Cityscapes dataset. Note (1) the users should
# register the Cityscapes dataset website at
# https://www.cityscapes-dataset.com/downloads/ to download the dataset,
# and (2) the users should download the utility scripts provided by
# Cityscapes at https://github.com/mcordts/cityscapesScripts.
#
# Usage:
#   bash ./preprocess_cityscapes.sh
#
# The folder structure is assumed to be:
#  + datasets
#    - build_cityscapes_data.py
#    - convert_cityscapes.sh
#    + cityscapes
#      + cityscapesscripts (downloaded scripts)
#      + gtFine
#      + leftImg8bit
#

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="."

# Root path for Cityscapes India dataset.
CITYSCAPES_ROOT="${WORK_DIR}/cityscapes_india"

#Create training labels.
python "${CURRENT_DIR}/cityscapes_india/cityscapesscripts/preparation/createTrainIdLabelImgs.py"


#Build TFRecords of the dataset.
#First, create output directory for storing TFRecords.
OUTPUT_DIR="${CITYSCAPES_ROOT}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

BUILD_SCRIPT="${CURRENT_DIR}/build_cityscapes_india_data.py"

echo "Converting Cityscapes dataset..."
python "${BUILD_SCRIPT}" \
  --cityscapes_root="${CITYSCAPES_ROOT}" \
  --output_dir="${OUTPUT_DIR}" \

echo "Creating the results on the input images"

INFERENCE_SCRIPT="${CURRENT_DIR}/vis.py"

python "${INFERENCE_SCRIPT}" \
  --logtostderr \
  --vis_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --vis_crop_size=1080 \
  --vis_crop_size=2048 \
  --max_number_of_evaluations=1 \
  --dataset="cityscapes_india" \
  --colormap_type="cityscapes_india" \
  --checkpoint_dir="${WORK_DIR}/cityscapes_india/exp/train_on_train_set/train" \
  --vis_logdir="${WORK_DIR}/cityscapes_india/exp/train_on_train_set/vis" \
  --dataset_dir="${WORK_DIR}/cityscapes_india/tfrecord"

# TRANSPARENT_SCRIPT="${CURRENT_DIR}/make_transparent_images_and_video.py"

# python "${TRANSPARENT_SCRIPT}" \
 # --cityscapes_india_root="${CURRENT_DIR}/cityscapes_india"

 
# 'C:/Users/pradeepr/Desktop/Movehack_2018/IIIT_H/models/research/deeplab/datasets/cityscapes_india/exp/train_on_train_set/train'
# 'C:/Users/pradeepr/Desktop/Movehack_2018/IIIT_H/models/research/deeplab/datasets/cityscapes_india/exp/train_on_train_set/vis'
# 'C:/Users/pradeepr/Desktop/Movehack_2018/IIIT_H/models/research/deeplab/datasets/cityscapes_india/tfrecord'
 