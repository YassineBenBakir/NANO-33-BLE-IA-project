#include <ArduTFLite.h>
#include "model_data.h"  // Modèle quantifié (mnist_cnn_quant_tflite)

// Configuration de la mémoire pour TensorFlow Lite
constexpr int tensorArenaSize = 8 * 1024;  // 8 Ko pour les tenseurs intermédiaires
alignas(16) byte tensorArena[tensorArenaSize];

// Buffers pour l’entrée
const int inputLength = 784;  // 28x28 pixels
float input_float[784];       // Réception série en float
int8_t input_int8[784];       // Entrée quantifiée en int8

void setup() {
  Serial.begin(115200);
  while (!Serial);

  // Message de démarrage
  Serial.println("Arduino prêt pour MNIST avec ArduTFLite");

  // Initialisation du modèle
  if (!modelInit(model, tensorArena, tensorArenaSize)) {
    Serial.println("Erreur : Initialisation du modèle échouée");
    while (true);
  }
  Serial.println("Modèle initialisé avec succès");
}

void loop() {
  if (Serial.available() > 0) {
    // Buffer pour la réception série
    char buffer[16384];
    int charsRead = 0;

    // Lire les données série jusqu’à la fin ou limite
    while (charsRead < 16383 && Serial.available()) {
      char c = Serial.read();
      if (c == '\n') break;
      buffer[charsRead++] = c;
    }
    buffer[charsRead] = '\0';

    Serial.print("Longueur reçue : ");
    Serial.println(charsRead);

    // Parser les 784 valeurs float
    char* token = strtok(buffer, ",");
    int i = 0;
    while (token != NULL && i < inputLength) {
      input_float[i] = atof(token);
      token = strtok(NULL, ",");
      i++;
    }

    Serial.print("Valeurs parsées : ");
    Serial.println(i);

    if (i == inputLength) {
      // Quantifier en int8 avec les paramètres du modèle (ajustez selon Python)
      float input_scale = 0.003921;  // Exemple, tiré de l’interpréteur TFLite
      int input_zero_point = -128;   // Exemple, tiré de l’interpréteur TFLite
      for (int j = 0; j < inputLength; j++) {
        input_int8[j] = (int8_t)(input_float[j] / input_scale + input_zero_point);
      }

      // Remplir le tenseur d’entrée
      for (int j = 0; j < inputLength; j++) {
        modelSetInput((float)input_int8[j], j);  // Conversion int8 vers float interne
      }

      // Exécuter l’inférence
      if (!modelRunInference()) {
        Serial.println("Erreur : Inférence échouée");
        return;
      }

      // Trouver la classe prédite (indice max)
      int max_idx = 0;
      float max_val = modelGetOutput(0);
      for (int k = 1; k < 10; k++) {
        float val = modelGetOutput(k);
        if (val > max_val) {
          max_val = val;
          max_idx = k;
        }
      }

      Serial.print("Prédiction : ");
      Serial.println(max_idx);
    } else {
      Serial.println("Erreur : Nombre de valeurs incorrect");
    }
  }
}