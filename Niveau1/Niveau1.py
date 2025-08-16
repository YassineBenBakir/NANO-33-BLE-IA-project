import numpy as np
from sklearn.neural_network import MLPClassifier

# Données fictives
X = np.array([[25, 40], [30, 80], [20, 60], [35, 20], [28, 90]])
y = np.array([0, 1, 0, 0, 1])

# Modèle avec 6000 neurones cachés
model = MLPClassifier(
    hidden_layer_sizes=(6000,),
    activation='relu',
    solver='adam',
    max_iter=2000,
    random_state=42
)

# Entraînement
model.fit(X, y)

# Tester
test_inputs = np.array([[30, 80], [35, 20]])
print("Prédictions :", model.predict(test_inputs))
print("Probabilités :", model.predict_proba(test_inputs))

# Extraire les poids et biais
weights_input_hidden = model.coefs_[0]    # 2x6000
weights_hidden_output = model.coefs_[1]   # 6000x1
bias_hidden = model.intercepts_[0]        # 6000
bias_output = model.intercepts_[1][0]     # Scalaire

# Générer weights_6000.h
with open("weights_6000.h", "w") as f:
    f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n#define HIDDEN_SIZE 6000\n\n")
    f.write("float w_input_hidden[2][HIDDEN_SIZE] = {\n")
    for i in range(2):
        f.write("  {" + ", ".join([f"{w:.8f}" for w in weights_input_hidden[i]]) + "}" + ("," if i < 1 else "") + "\n")
    f.write("};\n\n")
    f.write("float w_hidden_output[HIDDEN_SIZE] = {\n  ")
    f.write(", ".join([f"{w:.8f}" for w in weights_hidden_output.flatten()]))
    f.write("\n};\n\n")
    f.write("float b_hidden[HIDDEN_SIZE] = {\n  ")
    f.write(", ".join([f"{b:.8f}" for b in bias_hidden]))
    f.write("\n};\n\n")
    f.write(f"float b_output = {bias_output:.8f};\n\n#endif")

print("weights_6000.h généré avec succès !")