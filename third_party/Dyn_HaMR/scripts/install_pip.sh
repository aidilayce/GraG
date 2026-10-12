#!/usr/bin/env bash
set -e

echo "Creating virtual environment"
python3.10 -m venv .dynhamr
echo "Activating virtual environment"

source $PWD/.dynhamr/bin/activate

# install pytorch
$PWD/.dynhamr/bin/pip install torch==1.13.0 torchvision==0.14.0 --index-url https://download.pytorch.org/whl/cu117

# torch-scatter
$PWD/.dynhamr/bin/pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# install source
$PWD/.dynhamr/bin/pip install -e .

# install remaining requirements
$PWD/.dynhamr/bin/pip install -r requirements.txt

# install DROID-SLAM/DPVO
cd third-party/DROID-SLAM
python setup.py install
cd ../..

# install HaMeR
cd third-party/HaMeR
pip install -e .[all]
pip install -v -e third-party/ViTPose
