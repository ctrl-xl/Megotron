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
pip install ultralytics torch torchvision torchaudio scipy tensorflow onnx2tf
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