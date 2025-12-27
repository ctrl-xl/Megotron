import argparse
import os
from ultralytics import YOLO
import cv2
import numpy as np
import torch

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run YOLO prediction on an image')
    parser.add_argument('--image_name', type=str, default='test_images/test_close.jpg', help='Path to the image file (e.g., test_images/test_close.jpg)')
    parser.add_argument('--model', type=str, default='runs/detect/train/weights/best.pt', help='Path to YOLO model file')
    
    args = parser.parse_args()

    # --- DEBUT AJOUT ---
    print("-" * 30)
    if torch.cuda.is_available():
        print(f"✅ GPU DÉTECTÉ : {torch.cuda.get_device_name(0)}")
        processeur = 'cuda' 
    else:
        print("⚠️ ATTENTION : GPU non détecté, utilisation du CPU (Lent !)")
        processeur = 'cpu'
    print("-" * 30)
    # --- FIN AJOUT ---

    #processeur = 'cpu'
    # Use the image_name directly as the full path
    image_path = args.image_name
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    # Load the model
    try:
        model = YOLO(args.model)
        model.to(processeur)
        print(f"Loaded model: {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Predict with the model
    print(f"Running prediction on: {image_path}")
    results = model.predict(image_path)
    
    # Get the first result (since we're processing one image)
    if len(results) > 0:
        result = results[0]
        
        # Get prediction data
        if result.boxes is not None and len(result.boxes) > 0:
            xywh = result.boxes.xywh  # center-x, center-y, width, height
            xywhn = result.boxes.xywhn  # normalized
            xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
            xyxyn = result.boxes.xyxyn  # normalized
            names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
            confs = result.boxes.conf  # confidence score of each box
            
            print(f"Found {len(result.boxes)} objects:")
            for i, (name, conf) in enumerate(zip(names, confs)):
                print(f"  {i+1}. {name}: {conf:.3f} confidence")
        else:
            print("No objects detected in the image.")
        
        # Display the image with bounding boxes
        print("Displaying image with predictions...")
        result.show()  # This will display the image with boxes overlaid
        
        
    else:
        print("No results obtained from the model.")

if __name__ == "__main__":
    main()