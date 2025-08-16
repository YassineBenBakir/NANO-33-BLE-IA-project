import serial
import numpy as np
import time

# Charger une image de test
X_test = np.load("test_images.npy")
y_test = np.load("test_labels.npy")
test_image = X_test[0]

# Ouvrir la connexion série
ser = serial.Serial('COM5', 9600, timeout=5)  
ser.flush()


while "Arduino prêt pour MNIST" not in ser.readline().decode().strip():
    time.sleep(1)
print("Arduino détecté")

# Envoyer l’image
image_str = ','.join(map(str, test_image))
print(f"Longueur envoyée : {len(image_str)}")
ser.write((image_str + '\n').encode())
time.sleep(5)  

# Recevoir la réponse
while ser.in_waiting > 0:
    response = ser.readline().decode().strip()
    print(response)

print(f"Vraie classe : {y_test[0]}")
ser.close()