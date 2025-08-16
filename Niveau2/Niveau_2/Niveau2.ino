#include "weights_cnn.h"

float conv_output[26][26][CONV_FILTERS];
float pool_output[13][13][CONV_FILTERS];
float flatten_output[FLATTEN_SIZE];
float output[OUTPUT_SIZE];

float relu(float x) { return x > 0 ? x : 0; }

void conv_layer(float input[INPUT_HEIGHT][INPUT_WIDTH]) {
  for (int f = 0; f < CONV_FILTERS; f++) {
    for (int i = 0; i < 26; i++) {
      for (int j = 0; j < 26; j++) {
        conv_output[i][j][f] = conv_bias[f];
        for (int ki = 0; ki < CONV_KERNEL; ki++) {
          for (int kj = 0; kj < CONV_KERNEL; kj++) {
            conv_output[i][j][f] += input[i + ki][j + kj] * conv_weights[ki][kj][0][f];
          }
        }
        conv_output[i][j][f] = relu(conv_output[i][j][f]);
      }
    }
  }
}

void pool_layer() {
  for (int f = 0; f < CONV_FILTERS; f++) {
    for (int i = 0; i < 13; i++) {
      for (int j = 0; j < 13; j++) {
        float max_val = conv_output[i*2][j*2][f];
        for (int pi = 0; pi < POOL_SIZE; pi++) {
          for (int pj = 0; pj < POOL_SIZE; pj++) {
            float val = conv_output[i*2 + pi][j*2 + pj][f];
            if (val > max_val) max_val = val;
          }
        }
        pool_output[i][j][f] = max_val;
      }
    }
  }
}

void flatten_layer() {
  for (int i = 0; i < 13; i++) {
    for (int j = 0; j < 13; j++) {
      for (int f = 0; f < CONV_FILTERS; f++) {
        flatten_output[i * 13 * CONV_FILTERS + j * CONV_FILTERS + f] = pool_output[i][j][f];
      }
    }
  }
}

int dense_layer() {
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    output[i] = dense_bias[i];
    for (int j = 0; j < FLATTEN_SIZE; j++) {
      output[i] += flatten_output[j] * dense_weights[j][i];
    }
  }
  int max_idx = 0;
  for (int i = 1; i < OUTPUT_SIZE; i++) {
    if (output[i] > output[max_idx]) max_idx = i;
  }
  return max_idx;
}

void setup() {
  Serial.begin(115200);
  delay(2000);
  for (int i = 0; i < 3; i++) {
    Serial.println("Arduino prêt pour MNIST CNN");
    delay(500);
  }
}

void loop() {
  if (Serial.available() > 0) {
    float input[INPUT_HEIGHT][INPUT_WIDTH];
    char buffer[16384];
    int charsRead = 0;
    
    while (charsRead < 16383 && Serial.available()) {
      char c = Serial.read();
      if (c == '\n') break;
      buffer[charsRead++] = c;
    }
    buffer[charsRead] = '\0';
    
    Serial.print("Longueur reçue : ");
    Serial.println(charsRead);
    
    char* token = strtok(buffer, ",");
    int i = 0;
    for (int r = 0; r < INPUT_HEIGHT && token != NULL; r++) {
      for (int c = 0; c < INPUT_WIDTH && token != NULL; c++) {
        input[r][c] = atof(token);
        token = strtok(NULL, ",");
        i++;
      }
    }
    
    Serial.print("Valeurs parsées : ");
    Serial.println(i);
    
    if (i == INPUT_HEIGHT * INPUT_WIDTH) {
      conv_layer(input);
      pool_layer();
      flatten_layer();
      int prediction = dense_layer();
      Serial.print("Prédiction : ");
      Serial.println(prediction);
    } else {
      Serial.println("Erreur : nombre de valeurs incorrect");
    }
  }
}