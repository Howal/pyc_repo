#!/bin/bash

cd ./common/lib/bbox
python setup_linux.py build_ext --inplace
cd ../dataset/pycocotools
python setup_linux.py build_ext --inplace
cd ../../nms
python setup_linux.py build_ext --inplace
cd ../../operator_py
python setup_linux.py build_ext --inplace
cd ../..
#sudo /opt/conda/bin/pip install munkres

