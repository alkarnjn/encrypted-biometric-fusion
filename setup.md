### 

## Installation

SEAL 4.0.0 must be installed prior to compiling this project.
``
git clone --depth 1 https://github.com/alkarnjn/encrypted-feature-fusion.git
cd encrypted-feature-fusion/inference
- ("edit the CMakeLists.txt file to point to the SEAL 4.0.0 installation and change seal_shared to seal")
mkdir build; cd build
cmake ../
make clean; make
```

## create conda environment
```
conda create -n heft python=3.9 -y
conda activate heft
pip install -r requirements.txt
```