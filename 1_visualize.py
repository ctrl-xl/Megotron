from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2
import argparse
import os
import numpy as np
from supervision import Detections
import yaml

def load_yolo_labels(label_path, image_width, image_height):
    """Load YOLO format labels and convert to supervision Detections format"""
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return None
    
    boxes = []
    class_ids = []
    confidences = []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * image_width
                y_center = float(parts[2]) * image_height
                width = float(parts[3]) * image_width
                height = float(parts[4]) * image_height
                
                # Convert to xyxy format
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                boxes.append([x1, y1, x2, y2])
                class_ids.append(class_id)
                confidences.append(1.0)  # YOLO labels don't have confidence scores
    
    if not boxes:
        return None
    
    return Detections(
        xyxy=np.array(boxes),
        class_id=np.array(class_ids),
        confidence=np.array(confidences)
    )

def main():
    parser = argparse.ArgumentParser(description='Visualize object detection results')
    parser.add_argument('filename', nargs='?', help='Image filename to visualize (without path)')
    args = parser.parse_args()
    
    if args.filename:
        # Load pre-existing labeled data
        image_path = f"./source_images_labeled/train/images/{args.filename}"
        label_path = f"./source_images_labeled/train/labels/{args.filename.replace('.jpg', '.txt')}"
        
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return
        
        # Load image and get dimensions
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
        
        height, width = image.shape[:2]
        print(f"Loading image: {image_path}")
        print(f"Image dimensions: {width}x{height}")
        
        # Load labels
        detections = load_yolo_labels(label_path, width, height)
        if detections is None:
            print("No detections found in label file")
            return
        
        print(f"Loaded {len(detections)} detections from labels")
        
        # Load class names from data.yaml
        data_yaml_path = "./source_images_labeled/data.yaml"
        if os.path.exists(data_yaml_path):
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
                class_names = data_config.get('names', [])
        else:
            class_names = ['megot']  # fallback
        
        # Plot the results
        plot(
            image=image,
            classes=class_names,
            detections=detections
        )
        
    else:
        # Run model prediction (original behavior)
        model = GroundedSAM(ontology=CaptionOntology(
            {"cigarette butt": "megot", 
            "cigarette": "megot", 
            "cigarette filter": "megot", 
            "cigarette ash": "megot",
            "cigar butt": "megot",
            }))

        result = model.predict("./source_images/test_close.jpg")

        plot(
            image=cv2.imread("./source_images/test_close.jpg"),
            classes=model.ontology.classes(),
            detections=result
        )

if __name__ == "__main__":
    main()
