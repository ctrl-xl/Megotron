import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse
import torch
import sys

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run live YOLO prediction using camera feed')
    parser.add_argument('--model', type=str, default='runs/detect/train/weights/best.pt', 
                       help='Path to YOLO model file')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera device index (0 for default camera)')
    parser.add_argument('--confidence', type=float, default=0.5, 
                       help='Confidence threshold for detections')
    parser.add_argument('--fps', type=int, default=30, 
                       help='Target FPS for display')
    
    args = parser.parse_args()

    # --- Verification GPU ---
    print("-" * 30)
    device = 'cpu'
    if torch.cuda.is_available():
        print(f"✅ GPU DÉTECTÉ : {torch.cuda.get_device_name(0)}")
        device = 'cuda' 
    else:
        print("⚠️ ATTENTION : GPU non détecté, utilisation du CPU (Lent !)")
    print("-" * 30)

    # Initialisation variable cap à None pour sécurité
    cap = None
    
    try:
        # Load the custom trained YOLO model
        print(f"Loading model: {args.model}")
        model = YOLO(args.model)
        # Force le chargement initial pour éviter les lags après
        print("Model loaded successfully!")
    
        # Initialize camera INSIDE the try block
        print(f"Initializing camera (device {args.camera})...")
        cap = cv2.VideoCapture(args.camera)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera device {args.camera}")
            return
        
        # Configuration Camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        
        # Verification lecture réelle
        ret, test_frame = cap.read()
        if not ret:
            print("❌ Erreur critique : La caméra est ouverte mais n'envoie pas d'image.")
            return

        print(f"✅ Camera initialized successfully. Press 'q' to quit.")
        
        # Performance tracking
        fps_counter = 0
        fps_start_time = time.time()
        
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Perte du flux vidéo (Frame vide)")
                break
            
            # Run YOLO prediction
            results = model(frame, conf=args.confidence, verbose=False, device=device)
            
            # Process results
            annotated_frame = results[0].plot()
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                current_fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
                # Affiche FPS dans la console pour debug si l'écran fige
                # print(f"FPS: {current_fps:.1f}") 
            
            # Display info on frame
            cv2.putText(annotated_frame, f"FPS: {int(current_fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display
            cv2.imshow('Megotron Live Vision', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"capture_{timestamp}.jpg", annotated_frame)
                print(f"Saved capture_{timestamp}.jpg")
            
            # Control frame rate slightly to avoid heating if necessary
            # time.sleep(0.001) 
            
    except KeyboardInterrupt:
        print("\n⏹️ Arrêt par l'utilisateur (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE : {e}")
    finally:
        # C'est ici que la magie opère : Nettoyage GARANTI
        print("\n🧹 Nettoyage des ressources...")
        if cap is not None and cap.isOpened():
            cap.release()
            print("✅ Caméra libérée.")
        else:
            print("⚠️ Caméra déjà fermée ou non initialisée.")
            
        cv2.destroyAllWindows()
        print("Fin du programme.")

if __name__ == "__main__":
    main()