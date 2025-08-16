#include "weights_6000.h"

float hidden[HIDDEN_SIZE]; // Global pour éviter la pile

float relu(float x) { return x > 0 ? x : 0; }
float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

float predict(float input[2]) {
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    hidden[i] = b_hidden[i];
    for (int j = 0; j < 2; j++) {
      hidden[i] += input[j] * w_input_hidden[j][i];
    }
    hidden[i] = relu(hidden[i]);
  }
  float output = b_output;
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    output += hidden[i] * w_hidden_output[i];
  }
  return sigmoid(output);
}

void setup() {
  Serial.begin(9600);
  Serial.println("Début du programme"); // Débogage
  float input[2] = {30.0, 80.0};
  float result = predict(input);
  Serial.print("Prédiction statique : ");
  Serial.println(result, 4);
  Serial.println(result >= 0.5 ? "Classe : 1 (pluie)" : "Classe : 0 (pas pluie)");
}

void loop() {
  if (Serial.available() > 0) {
    float input[2];
    String data = Serial.readStringUntil('\n');
    Serial.print("Reçu : "); // Débogage
    Serial.println(data);
    sscanf(data.c_str(), "%f,%f", &input[0], &input[1]);
    float result = predict(input);
    Serial.print("Entrée : ");
    Serial.print(input[0]);
    Serial.print(", ");
    Serial.print(input[1]);
    Serial.print(" -> Prédiction : ");
    Serial.println(result, 4);
    Serial.println(result >= 0.5 ? "Classe : 1 (pluie)" : "Classe : 0 (pas pluie)");
  }
}