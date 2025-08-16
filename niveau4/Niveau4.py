import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np

# Charger et préparer MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Définir le CNN
model = models.Sequential([
    layers.Conv2D(4, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='valid'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

# Compiler et entraîner
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Évaluer le modèle flottant
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Précision test (float) : {test_acc:.4f}")

# Tester une image
test_image = x_test[0:1]
prediction = model.predict(test_image).argmax()
print(f"Prédiction Python (float) : {prediction}, Vraie classe : {y_test[0]}")

# Convertir en TensorFlow Lite avec quantification int8
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantification complète
converter.target_spec.supported_types = [tf.int8]      # Poids et activations en int8
converter.inference_input_type = tf.int8              # Entrée en int8
converter.inference_output_type = tf.int8             # Sortie en int8

# Fournir un jeu de données représentatif pour la calibration
def representative_dataset():
    for i in range(100):
        yield [x_train[i:i+1]]

converter.representative_dataset = representative_dataset
tflite_model = converter.convert()

# Sauvegarder le modèle TFLite
with open("mnist_cnn_quant.tflite", "wb") as f:
    f.write(tflite_model)

# Sauvegarder une image de test (non quantifiée pour l’instant)
np.save("test_image_quant.npy", x_test[0])
np.save("test_label_quant.npy", y_test[0])

print("Modèle quantifié sauvegardé comme mnist_cnn_quant.tflite")