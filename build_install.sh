#!/bin/bash

#python3 setup.py clean
#rm -fr build
#rm -fr dist
rm -fr dist/*
python3 setup.py build
python3 setup.py bdist_wheel
#pip3 uninstall talos -y
#rm -fr /home/viorelublea/venv/tf2/lib/python3.7/site-packages/talos
pip install --upgrade --force-reinstall --no-deps  dist/talos-0.6.4-py3-none-any.whl
