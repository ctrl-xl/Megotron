import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse
import torch
import sys

# --- Imports Matériel ---
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

# ==========================================
#        CONFIGURATION SERVOS (CONSTANTES)
# ==========================================
PCA_ADDR = 0x40 

# Mapping des ports
PIN_ROTATION   = 2   
PIN_HAUTEUR    = 1   
PIN_PROFONDEUR = 0   

# Angles de départ (HOME)
BASE_ANGLE_ROTATION   = 122.0  
BASE_ANGLE_HAUTEUR    = 10.0  
BASE_ANGLE_PROFONDEUR = 135.0  

# Limites de Sécurité
LIMITS_ROTATION   = (66, 180)   
LIMITS_HAUTEUR    = (10, 160)   
LIMITS_PROFONDEUR = (22, 170)   

# Paramètres de Tracking
CENTER_TOLERANCE = 7        
SLOW_THRESHOLD   = 30       # Seuil (pixels pour cam, degrés pour servos) pour ralentir

STEP_SIZE_FAST   = 1.0      # Vitesse "Approche" (Loin)
STEP_SIZE_SLOW   = 0.25     # Vitesse "Précision" (Proche)
CALIB_STEP       = 1.0      

# Etats du robot
ETAT_SEARCH  = 0  # Recherche / Alignement Rotation
ETAT_VERIFY  = 1  # Verrouillé -> Attente confirmation ESPACE
ETAT_DESCEND = 2  # Descente progressive vers la cible
ETAT_WAIT    = 3  # Attente en bas (Piquage)
ETAT_RESET   = 4  # Retour position Home
# ==========================================

def calculer_angles_depuis_pixels(pixel_y):
    """
    Calcule Hauteur et Profondeur basées sur la position Y du mégot dans l'image.
    """
    x = pixel_y
    # Formule Hauteur
    hauteur = (-0.00004636 * (x**2)) - (0.11793941 * x) + 173.86373626
    
    # Formule Profondeur
    profondeur = (-0.00015284 * (x**2)) + (0.04321251 * x) + 78.39481456
    
    return hauteur, profondeur

def deplacer_servo_progressif(current_angle, target_angle, servo_obj, limits):
    """
    Fonction utilitaire pour calculer le prochain pas d'un servo
    Renvoie: (nouvel_angle, est_arrive)
    """
    diff = target_angle - current_angle
    
    if abs(diff) < 0.5:
        return target_angle, True
    
    step = STEP_SIZE_FAST if abs(diff) > SLOW_THRESHOLD else STEP_SIZE_SLOW
    
    if diff > 0:
        new_angle = current_angle + step
        if new_angle > target_angle: new_angle = target_angle
    else:
        new_angle = current_angle - step
        if new_angle < target_angle: new_angle = target_angle
        
    min_v, max_v = limits
    new_angle = max(min_v, min(max_v, new_angle))
    
    if servo_obj:
        servo_obj.angle = new_angle
        
    return new_angle, False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='runs/detect/train/weights/best.pt', help='Chemin du modèle')
    parser.add_argument('--invert', action='store_true', help='Inverser la direction du servo')
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()

    # --- INITIALISATION MATÉRIEL ---
    print("-" * 30)
    print("INITIALISATION DU ROBOT...")
    
    pca = None
    axes = {
        'rotation':   {'servo': None, 'angle': BASE_ANGLE_ROTATION,   'label': "ROTATION",   'limits': LIMITS_ROTATION},
        'hauteur':    {'servo': None, 'angle': BASE_ANGLE_HAUTEUR,    'label': "HAUTEUR",    'limits': LIMITS_HAUTEUR},
        'profondeur': {'servo': None, 'angle': BASE_ANGLE_PROFONDEUR, 'label': "PROFONDEUR", 'limits': LIMITS_PROFONDEUR}
    }
    active_axis_key = 'rotation'

    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c, address=PCA_ADDR)
        pca.frequency = 50
        
        axes['rotation']['servo']   = servo.Servo(pca.channels[PIN_ROTATION])
        axes['hauteur']['servo']    = servo.Servo(pca.channels[PIN_HAUTEUR])
        axes['profondeur']['servo'] = servo.Servo(pca.channels[PIN_PROFONDEUR])
        
        print(f"Mise en position 'HOME'...")
        for key, axis in axes.items():
            if axis['servo']:
                min_v, max_v = axis['limits']
                safe_angle = max(min_v, min(max_v, axis['angle']))
                axis['angle'] = safe_angle
                axis['servo'].angle = safe_angle
        time.sleep(0.5)
        print("✅ Moteurs prêts.")

    except Exception as e:
        print(f"⚠️ ERREUR MATÉRIEL: {e}")

    # --- INITIALISATION VISION ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Démarrage Vision sur {device}...")
    try:
        model = YOLO(args.model)
    except:
        print("❌ Erreur chargement modèle.")
        return

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Erreur Caméra")
        return

    # =========================================================================
    # PHASE 1 : CALIBRATION
    # =========================================================================
    print("-" * 30)
    print("🖐️  MODE CALIBRATION ACTIVÉ")
    print("   SELECTION : [r] Rotation | [h] Hauteur | [p] Profondeur")
    print("   AJUSTEMENT: [<] GAUCHE   | [>] DROITE")
    print("   VALIDER   : [ESPACE]")
    print("-" * 30)

    calibration_done = False
    
    while not calibration_done:
        ret, frame = cap.read()
        if not ret: break
        
        annotated_frame = frame.copy()
        height, width = frame.shape[:2]
        center_x = width // 2
        
        # --- UI CALIBRATION ---
        cv2.line(annotated_frame, (center_x, 0), (center_x, height), (0, 255, 255), 1)
        cv2.putText(annotated_frame, "MODE CALIBRATION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Affichage des axes
        y_pos = 70
        for key, axis in axes.items():
            color = (0, 255, 0) if key == active_axis_key else (100, 100, 100)
            prefix = ">> " if key == active_axis_key else "   "
            min_v, max_v = axis['limits']
            text = f"{prefix}{axis['label']}: {axis['angle']:.1f} [{min_v}-{max_v}]"
            cv2.putText(annotated_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 30

        # Instructions en bas de l'écran
        cv2.putText(annotated_frame, "TOUCHES: [r]ot [h]aut [p]rof", (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(annotated_frame, "FLECHES: Ajuster | ESPACE: Valider", (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Megotron', annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'): 
            cap.release(); cv2.destroyAllWindows(); return
        elif key == 32: 
            calibration_done = True
        elif key == ord('r'): active_axis_key = 'rotation'
        elif key == ord('h'): active_axis_key = 'hauteur'
        elif key == ord('p'): active_axis_key = 'profondeur'
        elif key == 81 or key == ord('a'): # GAUCHE
            axis = axes[active_axis_key]
            if axis['servo']:
                axis['angle'] = max(axis['limits'][0], min(axis['limits'][1], axis['angle'] - CALIB_STEP))
                axis['servo'].angle = axis['angle']
        elif key == 83 or key == ord('d'): # DROITE
            axis = axes[active_axis_key]
            if axis['servo']:
                axis['angle'] = max(axis['limits'][0], min(axis['limits'][1], axis['angle'] + CALIB_STEP))
                axis['servo'].angle = axis['angle']

    # =========================================================================
    # PHASE 2 : AUTO (SEARCH -> VERIFY -> DESCEND -> WAIT -> RESET)
    # =========================================================================
    print("🚀 GO ! Mode AUTO actif.")
    
    robot_state = ETAT_SEARCH
    state_start_time = 0
    target_hauteur = 0
    target_profondeur = 0
    
    # Variables pour mémoriser la cible verrouillée (affichage)
    locked_box = None   # (x1, y1, x2, y2)
    locked_center = None # (cx, cy)

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            annotated_frame = frame.copy()
            height, width = frame.shape[:2]
            img_center_x = width // 2
            
            state_text = ["SEARCH", "VERIFY", "DESCEND", "WAIT", "RESET"][robot_state]
            if robot_state == ETAT_VERIFY: state_color = (0, 255, 255) # Jaune
            elif robot_state == ETAT_DESCEND: state_color = (255, 100, 0) # Bleu
            else: state_color = (0, 255, 0)
                
            cv2.putText(annotated_frame, f"ETAT: {state_text}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

            if robot_state == ETAT_SEARCH:
                cv2.line(annotated_frame, (img_center_x - CENTER_TOLERANCE, 0), (img_center_x - CENTER_TOLERANCE, height), (0, 255, 0), 1)
                cv2.line(annotated_frame, (img_center_x + CENTER_TOLERANCE, 0), (img_center_x + CENTER_TOLERANCE, height), (0, 255, 0), 1)

                results = model(frame, conf=0.3, verbose=False, device=device)
                detected_objects = []

                for box in results[0].boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    cx = int((coords[0] + coords[2]) / 2)
                    cy = int((coords[1] + coords[3]) / 2)
                    dist_msg = abs(cx - img_center_x) 
                    # On stocke aussi les coords brutes (x1, y1, x2, y2) pour l'affichage plus tard
                    detected_objects.append({'cx': cx, 'cy': cy, 'dist': dist_msg, 'box': coords})

                detected_objects.sort(key=lambda obj: obj['dist'])

                if len(detected_objects) > 0:
                    target = detected_objects[0]
                    tx, ty = target['cx'], target['cy']
                    error_x = tx - img_center_x
                    
                    if abs(error_x) > CENTER_TOLERANCE:
                        direction = -1 if args.invert else 1
                        step = STEP_SIZE_FAST if abs(error_x) > SLOW_THRESHOLD else STEP_SIZE_SLOW
                        
                        axes['rotation']['angle'] += (step * direction * (1 if error_x < 0 else -1))
                        
                        min_r, max_r = axes['rotation']['limits']
                        axes['rotation']['angle'] = max(min_r, min(max_r, axes['rotation']['angle']))
                        
                        if axes['rotation']['servo']:
                            axes['rotation']['servo'].angle = axes['rotation']['angle']
                        
                        cv2.line(annotated_frame, (img_center_x, int(height/2)), (tx, ty), (0, 255, 255), 2)
                    else:
                        # --- CIBLE TROUVÉE -> PASSAGE EN VÉRIFICATION ---
                        cv2.circle(annotated_frame, (tx, ty), 15, (0, 255, 0), 3)
                        
                        # Calcul immédiat des cibles finales
                        h_calc, p_calc = calculer_angles_depuis_pixels(ty)
                        target_hauteur = max(axes['hauteur']['limits'][0], min(axes['hauteur']['limits'][1], h_calc))
                        target_profondeur = max(axes['profondeur']['limits'][0], min(axes['profondeur']['limits'][1], p_calc))
                        
                        # --- SAUVEGARDE POUR AFFICHAGE ---
                        b = target['box']
                        locked_box = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                        locked_center = (tx, ty)

                        print(f"🔒 Cible verrouillée. H:{target_hauteur:.1f} P:{target_profondeur:.1f}")
                        robot_state = ETAT_VERIFY

            elif robot_state == ETAT_VERIFY:
                # --- AFFICHAGE DE LA CIBLE MÉMORISÉE ---
                if locked_box:
                    x1, y1, x2, y2 = locked_box
                    # Rectangle Vert autour du mégot
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if locked_center:
                    cx, cy = locked_center
                    # Point Rouge au centre
                    cv2.circle(annotated_frame, (cx, cy), 10, (0, 0, 255), -1)
                    # Texte coordonnées
                    cv2.putText(annotated_frame, f"CIBLE ({cx},{cy})", (cx + 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                cv2.putText(annotated_frame, "ESPACE: VALIDER | ESC: ANNULER", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"Cibles -> H:{target_hauteur:.0f} P:{target_profondeur:.0f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            elif robot_state == ETAT_DESCEND:
                curr_h = axes['hauteur']['angle']
                curr_p = axes['profondeur']['angle']
                new_h, done_h = deplacer_servo_progressif(curr_h, target_hauteur, axes['hauteur']['servo'], axes['hauteur']['limits'])
                new_p, done_p = deplacer_servo_progressif(curr_p, target_profondeur, axes['profondeur']['servo'], axes['profondeur']['limits'])
                axes['hauteur']['angle'] = new_h
                axes['profondeur']['angle'] = new_p
                
                diff_h = abs(target_hauteur - new_h)
                diff_p = abs(target_profondeur - new_p)
                cv2.putText(annotated_frame, f"Delta H:{diff_h:.1f} P:{diff_p:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if done_h and done_p:
                    print("⬇️ Descente terminée. Piquage.")
                    state_start_time = time.time()
                    robot_state = ETAT_WAIT

            elif robot_state == ETAT_WAIT:
                elapsed = time.time() - state_start_time
                remaining = 2.0 - elapsed
                cv2.putText(annotated_frame, f"PIQUAGE: {remaining:.1f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if elapsed >= 2.0:
                    robot_state = ETAT_RESET

            elif robot_state == ETAT_RESET:
                cv2.putText(annotated_frame, "RETOUR HOME...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
                if axes['hauteur']['servo']: 
                    axes['hauteur']['servo'].angle = BASE_ANGLE_HAUTEUR
                    axes['hauteur']['angle'] = BASE_ANGLE_HAUTEUR
                    time.sleep(0.1)
                if axes['profondeur']['servo']: 
                    axes['profondeur']['servo'].angle = BASE_ANGLE_PROFONDEUR
                    axes['profondeur']['angle'] = BASE_ANGLE_PROFONDEUR
                    time.sleep(0.1)
                time.sleep(1.0)
                print("🔄 Prêt pour le prochain.")
                
                # Réinitialisation des variables de cible pour le prochain cycle
                locked_box = None
                locked_center = None
                robot_state = ETAT_SEARCH

            cv2.imshow('Megotron', annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == 27: # ESC (Code 27)
                if robot_state == ETAT_VERIFY:
                    print("↩️ Verrouillage annulé. Retour Recherche.")
                    locked_box = None
                    locked_center = None
                    robot_state = ETAT_SEARCH

            elif key == 32: # ESPACE
                if robot_state == ETAT_VERIFY:
                    print("🚀 Validation reçue.")
                    robot_state = ETAT_DESCEND

    except KeyboardInterrupt:
        print("\nArrêt demandé...")
    
    finally:
        print("\n🛑 NETTOYAGE...")
        if pca:
            try:
                pca.channels[PIN_ROTATION].duty_cycle = 0
                pca.channels[PIN_HAUTEUR].duty_cycle = 0
                pca.channels[PIN_PROFONDEUR].duty_cycle = 0
                pca.deinit()
            except: pass
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()