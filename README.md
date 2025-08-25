# Megotron
A robot who collects cigarettes using an AI detection program to identify them


## Installation Guide
### Install the ultralytics package from PyPI
``` bash
source venv/bin/activate
pip install ultralytics
pip install torch torchvision torchaudio
```

## Autodistill install

See  [Auto distill doc](https://docs.autodistill.com/)
and  [Auto distill target model yolov11](https://github.com/autodistill/autodistill-yolov11)

``` bash
pip install autodistill autodistill-grounding-dino roboflow scikit-learn 
```

## Files usages
### Label the dataset
- [0_predict_dataset.py](0_predict_dataset.py): Creates the coordinates for the bounding boxes around the cigarettes within 'labels'
```bash
python3 predict_dataset.py
```

### Visualize the labeling
- [1_visualize.py](1_visualize.py): Draws the bounding box over the picture
```bash
python3 visualize.py image_name.jpg
```