import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np

# Charger et préparer MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0  # Normaliser
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)  # Ajouter canal
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

# Évaluer
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Précision test : {test_acc:.4f}")

# Tester une image
test_image = x_test[0:1]  # Batch de 1
prediction = model.predict(test_image).argmax()
print(f"Prédiction Python : {prediction}, Vraie classe : {y_test[0]}")

# Exporter les poids
conv_weights, conv_bias = model.layers[0].get_weights()  # Conv2D : 3x3x1x4
dense_weights, dense_bias = model.layers[3].get_weights()  # Dense : 676x10

# Générer weights_cnn.h
with open("weights_cnn.h", "w") as f:
    f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")
    f.write("#define INPUT_HEIGHT 28\n#define INPUT_WIDTH 28\n#define CONV_FILTERS 4\n")
    f.write("#define CONV_KERNEL 3\n#define POOL_SIZE 2\n")
    f.write("#define FLATTEN_SIZE 676\n#define OUTPUT_SIZE 10\n\n")
    
    # Poids convolutionnels (3x3x1x4)
    f.write("float conv_weights[CONV_KERNEL][CONV_KERNEL][1][CONV_FILTERS] = {\n")
    for i in range(3):
        f.write("  {\n")
        for j in range(3):
            f.write("    {{" + ", ".join([f"{conv_weights[i,j,0,k]:.8f}" for k in range(4)]) + "}}" + ("," if j < 2 else "") + "\n")
        f.write("  }" + ("," if i < 2 else "") + "\n")
    f.write("};\n\n")
    
    f.write("float conv_bias[CONV_FILTERS] = {" + ", ".join([f"{b:.8f}" for b in conv_bias]) + "};\n\n")
    
    # Poids Dense (676x10)
    f.write("float dense_weights[FLATTEN_SIZE][OUTPUT_SIZE] = {\n")
    for i in range(676):
        f.write("  {" + ", ".join([f"{dense_weights[i,j]:.8f}" for j in range(10)]) + "}" + ("," if i < 675 else "") + "\n")
    f.write("};\n\n")
    
    f.write("float dense_bias[OUTPUT_SIZE] = {" + ", ".join([f"{b:.8f}" for b in dense_bias]) + "};\n\n")
    f.write("#endif")

# Sauvegarder une image de test
np.save("test_image_cnn.npy", x_test[0])
np.save("test_label_cnn.npy", y_test[0])

print("weights_cnn.h et fichiers de test générés !")