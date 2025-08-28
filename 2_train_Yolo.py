from autodistill_yolov11 import YOLOv11
model = YOLOv11("yolo11n.pt")
model.train("./source_images_labeled/data.yaml", epochs=100)
