import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run live YOLO prediction using camera feed')
    parser.add_argument('--model', type=str, default='runs/detect/train/weights/best.pt', 
                       help='Path to YOLO model file')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera device index (0 for default camera)')
    parser.add_argument('--confidence', type=float, default=0.1, 
                       help='Confidence threshold for detections')
    parser.add_argument('--fps', type=int, default=30, 
                       help='Target FPS for display')
    
    args = parser.parse_args()
    
    # Load the custom trained YOLO model
    try:
        print(f"Loading model: {args.model}")
        model = YOLO(args.model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize camera
    print(f"Initializing camera (device {args.camera})...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera device {args.camera}")
        return
    
    # Set camera properties for better performance on Mac
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    # Get camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera initialized: {frame_width}x{frame_height} @ {fps:.1f} FPS")
    print("Press 'q' to quit, 's' to save current frame")
    
    # Performance tracking
    frame_count = 0
    start_time = time.time()
    fps_counter = 0
    fps_start_time = time.time()
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Run YOLO prediction
            results = model(frame, conf=args.confidence, verbose=False)
            
            # Process results and draw on frame
            annotated_frame = results[0].plot()
            
            # Calculate and display FPS
            frame_count += 1
            fps_counter += 1
            
            if time.time() - fps_start_time >= 1.0:
                current_fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            else:
                current_fps = 0
            
            # Add FPS and model info to frame
            cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Model: {args.model.split('/')[-1]}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Conf: {args.confidence}", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display detection count
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                detection_count = len(results[0].boxes)
                cv2.putText(annotated_frame, f"Detections: {detection_count}", 
                           (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display detection details
                for i, (box, conf, cls) in enumerate(zip(results[0].boxes.xyxy, 
                                                        results[0].boxes.conf, 
                                                        results[0].boxes.cls)):
                    if i < 3:  # Show first 3 detections
                        class_name = results[0].names[int(cls)]
                        cv2.putText(annotated_frame, f"{class_name}: {conf:.2f}", 
                                   (10, 160 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, (0, 255, 0), 2)
            else:
                cv2.putText(annotated_frame, "No detections", 
                           (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Display the frame
            cv2.imshow('Live YOLO Detection', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"live_capture_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Saved frame as: {filename}")
            
            # Control frame rate
            time.sleep(1.0 / args.fps)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # Print performance summary
        elapsed_time = time.time() - start_time
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        print(f"\nPerformance Summary:")
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    main()
