#!/bin/bash

source /global/project/projectdirs/m1092/xju/miniconda3/bin/activate py3.6
export LD_LIBRARY_PATH=/global/homes/x/xju/project/xju/miniconda3/envs/py3.6/lib:$LD_LIBRARY_PATH

python trainSegmentClassifier.py --input-dir kaggle --output-dir model2 --n-samples 1768 --n-epochs 10
