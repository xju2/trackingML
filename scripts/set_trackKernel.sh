#!/bin/bash
module load gcc/7.3.0
module load cmake/3.8.2

source /global/project/projectdirs/m1092/xju/miniconda3/bin/activate py3.6
export LD_LIBRARY_PATH=/global/homes/x/xju/project/xju/miniconda3/envs/py3.6/lib:$LD_LIBRARY_PATH

python -m ipykernel_launcher $@
