# Broche 1 (jaune) (3.3V) : Alimentation logique des modules.
# Broche 3 (orange) (SDA) : Ligne de données (Serial Data).
# Broche 5 (rouge) (SCL) : Ligne d'horloge (Serial Clock).
# Broche 6 (brun) (GND) : Terre commune (Ground).


import time
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

# Configuration de l'I2C
# Note : Sur le Jetson Xavier, le bus 8 est souvent mappé sur les pins par défaut.
# Si board.SCL/SDA ne fonctionnent pas, il faudra peut-être spécifier le bus manuellement.
i2c = busio.I2C(board.SCL, board.SDA)

# Initialisation du PCA9685 à l'adresse 0x70
pca = PCA9685(i2c, address=0x40)
pca.frequency = 50  # Les servos attendent généralement 50Hz

# Création de l'objet servo sur le canal 0
# Assure-toi que ton servo est branché sur la prise 0 du driver
servo0 = servo.Servo(pca.channels[0])
servo1 = servo.Servo(pca.channels[1])

print("Début du test servo...")

try:
    servo1.angle = 110
    while True:
        print("Angle: 15 degrés")
        servo0.angle = 15
        time.sleep(1)
        
        print("Angle: 30 degrés")
        servo0.angle = 30
        time.sleep(1)

        print("Angle: 45 degrés")
        servo0.angle = 45
        time.sleep(1)

        print("Angle: 30 degrés")
        servo0.angle = 30
        time.sleep(1)

except KeyboardInterrupt:
    print("\nArrêt par l'utilisateur.")

except Exception as e:
    print(f"\nUne erreur est survenue : {e}")

finally:
    # Ce bloc s'exécute TOUJOURS, même en cas de crash
    print("Nettoyage des ressources...")
    
    # 1. On coupe le moteur pour qu'il ne force pas
    pca.channels[0].duty_cycle = 0
    
    # 2. On libère le PCA9685 et le bus I2C
    pca.deinit()
    # Si i2c a été créé indépendamment et n'est pas géré par pca.deinit() :
    i2c.deinit() 
    
    print("Terminé.")