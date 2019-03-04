#!/bin/bash
ls
cd ./common/lib/bbox
/opt/conda/bin/python setup_linux.py build_ext --inplace
cd ../dataset/pycocotools
/opt/conda/bin/python setup_linux.py build_ext --inplace
cd ../../nms
/opt/conda/bin/python setup_linux.py build_ext --inplace
cd ../..
/opt/conda/bin/pip install munkres==1.0.12
