import serial
import numpy as np
import time

# Charger l’image de test
test_image = np.load("test_image_cnn.npy")
test_label = np.load("test_label_cnn.npy")

# Ouvrir la connexion série
ser = serial.Serial('COM5', 115200, timeout=2)  
ser.flush()

# Attendre la synchronisation
time.sleep(2)
for _ in range(5):
    if ser.in_waiting > 0:
        print(ser.readline().decode().strip())
    time.sleep(1)

# Envoyer l’image (784 valeurs)
image_str = ','.join(map(str, test_image.flatten()))
print(f"Longueur envoyée : {len(image_str)}")
ser.write((image_str + '\n').encode())
time.sleep(5)

# Recevoir la réponse
while ser.in_waiting > 0:
    response = ser.readline().decode().strip()
    print(response)

print(f"Vraie classe : {test_label}")
ser.close()