# Megotron
A robot who collects cigarettes using an AI detection program to identify them


## Installation Guide
### Create a python virtual environment (if none exist)
``` bash
brew install python
python3 -m venv venv
```

### Install the ultralytics package from PyPI
``` bash
source venv/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
source ~/.bashrc
sudo apt-get update
sudo apt-get install -y libopenblas-base libopenmpi-dev libomp-dev libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev

pip install numpy
pip install --no-cache-dir https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

git clone --branch v0.16.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.16.1
python3 setup.py install --user
pip install pybind11 setuptools wheel "pillow<11" urllib3 idna certifi
cd ..


# Install dependencies
pip install -U pip six numpy wheel setuptools mock 'future>=0.17.1' 'gast==0.4.0' 'protobuf<3.20' pybind11 cython pkgconfig packaging h5py

sudo apt-get install -y liblapack-dev libblas-dev gfortran

# Install TensorFlow (compatible with JetPack 5)
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v51 tensorflow==2.11.0+nv23.01


pip install ultralytics scipy onnx2tf

# Install the base wrapper
pip install autodistill

# Install Grounding DINO - this will trigger a compilation of 'groundingdino'
# We set MAX_JOBS=4 to prevent the Xavier from running out of RAM during compilation
export MAX_JOBS=4
pip install autodistill-grounding-dino

```

## Autodistill install

See  [Auto distill doc](https://docs.autodistill.com/)
and  [Auto distill target model yolov11](https://github.com/autodistill/autodistill-yolov11)

``` bash
pip install autodistill autodistill-grounding-dino roboflow scikit-learn autodistill-yolov11
```

## Files usages
### Label the dataset from the source_images/ folder
- [0_predict_dataset.py](0_predict_dataset.py): Creates the coordinates for the bounding boxes around the cigarettes within 'labels'
```bash
python3 0_predict_dataset.py
```

### Visualize the labeling
- [1_visualize.py](1_visualize.py): Draws the bounding box over the picture
```bash
python3 1_visualize.py image_name.jpg
```

###Train the model using the dataset
- [2_train_Yolo.py](2_train_Yolo.py): Training the AI with the images from [source_images](/Users/lyracoupez/Documents/megotron/Megotron/source_images) during 100 epochs
```bash
python3 2_train_Yolo.py
```

###Validate the finetuned model
- [3_validate_yolo.py](3_validate_yolo.py): Validating the finetuning with test images from the dataset 
```bash
python3 3_validate_yolo.py
```

###Exporting the model to have it on a smaller device
-[4_export](4_export_to_tensorflow_lite.py)