#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate CVLab-license-plate

export PYTHONPATH=$PWD:$PWD/..

python web_demo.py
