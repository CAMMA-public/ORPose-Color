#!/bin/bash
# coding: utf-8
'''
Copyright University of Strasbourg, All Rights Reserved.
'''
cd ..
source $(conda info --base)/bin/activate
conda activate pt

# Paths and params
SCALE=12 # or 8, 12

MODEL_WEIGHTS="models/orpose_fixed_${SCALE}x_model_final.pth"
CONFIG_FILE="configs/orpose_fixed_${SCALE}x.yaml"

H36_STATS_PATH="data/h36stats.json"
TEST_SET_JSON="datasets/mvor/annotations/camma_mvor_2019_color.json"
TEST_SET_IMGDIR="datasets/mvor/images"

python eval_net.py --config-file  ${CONFIG_FILE} \
                    --eval-only \
                    MODEL.WEIGHTS ${MODEL_WEIGHTS} \
                    DATASETS.H36_STATS_PATH ${H36_STATS_PATH} \
                    DATASETS.TEST_SET_NAME "mvor" \
                    DATASETS.TEST_SET_JSON ${TEST_SET_JSON} \
                    DATASETS.TEST_SET_IMGDIR ${TEST_SET_IMGDIR} \
                    DATASETS.DOWN_SAMPLING_FIX_SCALE ${SCALE}.0 \
                    DATASETS.FIX_DOWN_SAMPLING True \
