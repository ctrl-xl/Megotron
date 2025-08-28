from ultralytics import YOLO

# Load your custom-trained YOLOv10 model
# Replace 'path/to/your/best.pt' with the actual path to your model file.
model = YOLO('runs/detect/train/weights/best.pt')

# Export the model to TFLite format
# The export will create a file named 'best.tflite'
# You can also specify the image size with imgsz=640 for example
model.export(format='tflite')
