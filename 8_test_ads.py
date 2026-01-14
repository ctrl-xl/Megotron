import time
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

from adafruit_ads1x15 import ADS1115, AnalogIn, ads1x15

#import adafruit_ads1x15.ads1115 as ADS
#from adafruit_ads1x15.analog_in import AnalogIn

# --- Configuration ---
PCA_ADDR = 0x40  # Adresse du driver servo (Vérifiée)
ADS_ADDR = 0x48  # Adresse du convertisseur ADC (Vérifiée)

print("Initialisation du Bus I2C via 'board'...")

try:
    # On utilise la méthode standard. 
    # Sur Jetson Xavier, board.SCL et board.SDA correspondent généralement 
    # aux pins physiques 5 et 3, qui sont reliées au Bus 8.
    i2c = busio.I2C(board.SCL, board.SDA)
    
    # 1. Configuration du PCA9685 (Servos)
    pca = PCA9685(i2c, address=PCA_ADDR)
    pca.frequency = 50
    
    # Servos sur les canaux 0 et 1
    servo0 = servo.Servo(pca.channels[0])
    # servo1 = servo.Servo(pca.channels[1]) # Décommenter si utilisé

    # 2. Configuration du ADS1115 (Capteurs/Retour position)
    ads = ADS1115(i2c, address=ADS_ADDR)
    ads.gain = 1 

    # Création de l'entrée analogique sur la broche A0 du module ADS
    feedback_channel = AnalogIn(ads, ads1x15.Pin.A0)

    print("Démarrage. Lecture du capteur sur A0.")
    print(f" -> Lecture ADC: {feedback_channel.value} (Tension: {feedback_channel.voltage:.2f}V)")
    time.sleep(5)
        
    while True:
        # Mouvement 1 : 15 degrés
        print(f"\nCommande Servo: 15°")
        servo0.angle = 15
        time.sleep(0.5)
        # Lecture de la tension
        # value = brut (0-32767), voltage = tension réelle
        print(f" -> Lecture ADC: {feedback_channel.value} (Tension: {feedback_channel.voltage:.2f}V)")
        time.sleep(5)
        
        # Mouvement 2 : 45 degrés
        print(f"\nCommande Servo: 45°")
        servo0.angle = 45
        time.sleep(0.5)
        print(f" -> Lecture ADC: {feedback_channel.value} (Tension: {feedback_channel.voltage:.2f}V)")
        time.sleep(5)

except KeyboardInterrupt:
    print("\nArrêt demandé.")

except ValueError as e:
    print(f"\nErreur I2C/Valeur : {e}")
    print("Vérifie que board.SCL et SDA pointent bien vers le Bus 8.")

except Exception as e:
    print(f"\nErreur : {e}")

finally:
    # Nettoyage
    if 'pca' in locals():
        pca.channels[0].duty_cycle = 0
        pca.deinit()
    # Si i2c a été créé, on peut essayer de le fermer, 
    # mais busio.I2C ne supporte pas toujours deinit() proprement sur toutes les plateformes
    # si utilisé hors d'un contexte 'with'.
    print("Bus libéré.")