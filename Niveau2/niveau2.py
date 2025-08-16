import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import pickle

# Charger MNIST
print("Chargement de MNIST...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0  # Normaliser entre 0 et 1
y = y.astype(np.uint8)

# Split entraînement/test
X_train, y_train = X[:60000], y[:60000]
X_test, y_test = X[60000:], y[60000:]

# Standardisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modèle avec 32 neurones cachés
print("Entraînement du modèle...")
model = MLPClassifier(
    hidden_layer_sizes=(32,),
    activation='relu',
    solver='adam',
    max_iter=20,
    random_state=42
)
model.fit(X_train, y_train)

# Évaluer
print(f"Précision entraînement : {model.score(X_train, y_train):.4f}")
print(f"Précision test : {model.score(X_test, y_test):.4f}")

# Tester une image
test_image = X_test[0]
prediction = model.predict([test_image])[0]
print(f"Prédiction pour la première image : {prediction}, Vraie classe : {y_test[0]}")

# Exporter les poids
weights_input_hidden = model.coefs_[0]    # 784x32
weights_hidden_output = model.coefs_[1]   # 32x10
bias_hidden = model.intercepts_[0]        # 32
bias_output = model.intercepts_[1]        # 10

# Générer weights_mnist.h
with open("weights_mnist.h", "w") as f:
    f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")
    f.write("#define INPUT_SIZE 784\n#define HIDDEN_SIZE 32\n#define OUTPUT_SIZE 10\n\n")
    
    f.write("float w_input_hidden[INPUT_SIZE][HIDDEN_SIZE] = {\n")
    for i in range(784):
        f.write("  {" + ", ".join([f"{w:.8f}" for w in weights_input_hidden[i]]) + "}" + ("," if i < 783 else "") + "\n")
    f.write("};\n\n")
    
    f.write("float w_hidden_output[HIDDEN_SIZE][OUTPUT_SIZE] = {\n")
    for i in range(32):
        f.write("  {" + ", ".join([f"{w:.8f}" for w in weights_hidden_output[i]]) + "}" + ("," if i < 31 else "") + "\n")
    f.write("};\n\n")
    
    f.write("float b_hidden[HIDDEN_SIZE] = {\n  ")
    f.write(", ".join([f"{b:.8f}" for b in bias_hidden]))
    f.write("\n};\n\n")
    
    f.write("float b_output[OUTPUT_SIZE] = {\n  ")
    f.write(", ".join([f"{b:.8f}" for b in bias_output]))
    f.write("\n};\n\n")
    f.write("#endif")

# Sauvegarder pour tests
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
np.save("test_images.npy", X_test[:10])
np.save("test_labels.npy", y_test[:10])

print("weights_mnist.h et fichiers de test générés !")