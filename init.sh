#!/bin/bash

cd ./common/lib/bbox
python setup_linux.pyc build_ext --inplace
cd ../dataset/pycocotools
python setup_linux.pyc build_ext --inplace
cd ../../nms
python setup_linux.pyc build_ext --inplace
cd ../..
sudo /opt/conda/bin/pip install munkres
