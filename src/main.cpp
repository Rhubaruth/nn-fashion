#include "layers.h"
#include "load_data.h"
#include <cmath>
#include <iostream>

#include <algorithm>

// ---------------------------------------------------------------------------
// NETWORK LAYERS
// ---------------------------------------------------------------------------
unsigned seed = 1;
DenseLayer hidden1(784, 64, Activation::LeakyReLU, seed);
DenseLayer hidden2(64, 64, Activation::LeakyReLU, seed);
DenseLayer output(64, 10, Activation::Linear, seed);

// ---------------------------------------------------------------------------
// TRAINING
// ---------------------------------------------------------------------------
const double lr = 0.001;
const size_t numEpochs = 20;
const size_t BATCH_SIZE = 8;

const double DECAY_RATE1 = 0.9;
const double DECAY_RATE2 = 0.999;

const double DELTA = pow(10, -8);

// ---------------------------------------------------------------------------

std::vector<double> softmax(const std::vector<double> &vec) {
  double max = vec[0];
  for (double i : vec) {
    if (i > max)
      max = i;
  }

  double sum = 0.0;
  std::vector<double> output(vec.size(), 0);
  for (unsigned int i = 0; i < vec.size(); i++) {
    output[i] = exp(vec[i] - max); // subtract max for numeric stability
    sum += output[i];
  }
  for (unsigned int i = 0; i < vec.size(); i++) {
    output[i] /= sum;
  }
  return output;
}

size_t argmax(const std::vector<double> &v) {
  return std::max_element(v.begin(), v.end()) - v.begin();
}

// ---------------------------------------------------------------------------

// RMSProp table
std::vector<double> RMSPropW[3] = {
    std::vector(hidden1.getWeights().size(), 0.0),
    std::vector(hidden2.getWeights().size(), 0.0),
    std::vector(output.getWeights().size(), 0.0),
};
std::vector<double> RMSPropB[3] = {
    std::vector(hidden1.getBiases().size(), 0.0),
    std::vector(hidden2.getBiases().size(), 0.0),
    std::vector(output.getBiases().size(), 0.0),
};
// Momentum Table
std::vector<double> momentumW[3] = {
    std::vector(hidden1.getWeights().size(), 0.0),
    std::vector(hidden2.getWeights().size(), 0.0),
    std::vector(output.getWeights().size(), 0.0),
};
std::vector<double> momentumB[3] = {
    std::vector(hidden1.getBiases().size(), 0.0),
    std::vector(hidden2.getBiases().size(), 0.0),
    std::vector(output.getBiases().size(), 0.0),
};

// ---------------------------------------------------------------------------

void network_inference(const std::vector<double> &input) {
  // for (size_t i = 0; i < input.size(); i++) {
  //   input[i] = input[i] / 255.0;
  // }
  hidden1.forward(input);
  hidden2.forward(hidden1.getValues());
  output.forward(hidden2.getValues());
}

void network_backprop(const std::vector<double> &input, double target) {
  std::vector<double> out_vec = softmax(output.getValues());
  size_t out_size = output.getValues().size();
  std::vector<double> wanted(out_size, 0);
  wanted[target] = 1;

  std::vector<double> dw(out_size, 0);
  for (size_t i = 0; i < out_size; i++) {
    dw[i] = out_vec[i] - wanted[i];
  }

  output.backward(dw, hidden2.getValues());
  hidden2.backward(output, hidden1.getValues());
  hidden1.backward(hidden2, input);
}

void SGD(DenseLayer &layer, double lr, size_t sample_size) {
  if (sample_size == 0) {
    printf("[ERROR] SDG got sample_size=0!");
    return;
  }
  std::vector<double> &W = layer.getWeights();
  std::vector<double> &B = layer.getBiases();
  const std::vector<double> &gW = layer.getGradWeights();
  const std::vector<double> &gB = layer.getGradBiases();

  double scale_lr = lr / double(sample_size);
  for (size_t i = 0; i < W.size(); i++) {
    W[i] -= gW[i] * scale_lr;
  }
  for (size_t i = 0; i < B.size(); i++) {
    B[i] -= gB[i] * scale_lr;
  }
  layer.zeroGradients();
}

void Adam(DenseLayer &layer, double lr, size_t sample_size, size_t j,
          size_t t) {
  std::vector<double> &W = layer.getWeights();
  std::vector<double> &B = layer.getBiases();
  const std::vector<double> &gW = layer.getGradWeights();
  const std::vector<double> &gB = layer.getGradBiases();

  const double scale_lr = lr / double(sample_size);

  for (size_t i = 0; i < W.size(); i++) {
    RMSPropW[j][i] =
        DECAY_RATE2 * RMSPropW[j][i] + (1 - DECAY_RATE2) * pow(gW[i], 2);
    momentumW[j][i] = DECAY_RATE1 * momentumW[j][i] + (1 - DECAY_RATE1) * gW[i];

    double mHat = momentumW[j][i] / (1 - pow(DECAY_RATE1, t));
    double rHat = RMSPropW[j][i] / (1 - pow(DECAY_RATE2, t));
    W[i] -= scale_lr / std::sqrt(rHat + DELTA) * mHat;
  }
  for (size_t i = 0; i < B.size(); i++) {
    RMSPropB[j][i] =
        DECAY_RATE2 * RMSPropB[j][i] + (1 - DECAY_RATE2) * pow(gB[i], 2);
    momentumB[j][i] = DECAY_RATE1 * momentumB[j][i] + (1 - DECAY_RATE1) * gB[i];

    double mHat = momentumB[j][i] / (1 - pow(DECAY_RATE1, t));
    double rHat = RMSPropB[j][i] / (1 - pow(DECAY_RATE2, t));
    B[i] -= scale_lr / std::sqrt(rHat + DELTA) * mHat;
  }

  layer.zeroGradients();
}

// OUTPUT
void write_output(const std::string filename,
                  const std::vector<std::vector<double>> &inputs) {
  std::ofstream file(filename);
  if (!file)
    throw std::runtime_error("Could not open file: " + filename);

  for (size_t i = 0; i < inputs.size(); i++) {
    network_inference(inputs[i]);
    size_t prediction = argmax(softmax(output.getValues()));
    file << prediction << "\n";
  }
  file.close();
}

// MAIN LOOP
int main() {

  std::vector<std::vector<double>> trainingInputs =
      read_csv("data/fashion_mnist_train_vectors.csv", 784);
  std::vector<std::vector<double>> trainingLabels =
      read_csv("data/fashion_mnist_train_labels.csv", 1);

  // Normalize inputs
  for (auto &vec : trainingInputs) {
    for (auto &t : vec) {
      t = t / 255.0;
    }
  }

  output.zeroGradients();
  hidden1.zeroGradients();
  hidden2.zeroGradients();

  size_t batchCount = 0;
  // int correct = 0;

  printf("---   PARAMETERS   ---\n");
  printf("Seed: %d\n\n", seed);
  printf("Layer1 size =%lu\n", hidden1.getNumNeurons());
  printf("Layer2 size =%lu\n", hidden2.getNumNeurons());
  printf("Layer3 size =%lu\n", output.getNumNeurons());
  printf("decay Momentum Beta1=%g\n", DECAY_RATE1);
  printf("decay RMSProp  Beta2=%g\n", DECAY_RATE2);
  printf("\nLearningRate lr=%g\n", lr);
  printf("numEpochs =%lu\n", numEpochs);
  printf("batchSize =%lu\n", BATCH_SIZE);

  // Train neural network
  printf("\nTRAINING: \n");
  size_t numTrainingSets = trainingInputs.size();

  size_t t = 1; // Number of updates for Bias correction
  for (unsigned int epoch = 0; epoch < numEpochs; epoch++) {
    for (size_t x = 0; x < numTrainingSets; x++) {
      batchCount++;
      network_inference(trainingInputs[x]);

      // Backpropagation
      network_backprop(trainingInputs[x], trainingLabels[x][0]);

      if (batchCount == BATCH_SIZE) {
        Adam(hidden1, lr, batchCount, 0, t);
        Adam(hidden2, lr, batchCount, 1, t);
        Adam(output, lr, batchCount, 2, t);

        t += 1;
        batchCount = 0;
      }
    }
  }

  if (batchCount > 0) {
    // SGD
    Adam(hidden1, lr, batchCount, 0, t);
    Adam(hidden2, lr, batchCount, 1, t);
    Adam(output, lr, batchCount, 2, t);

    batchCount = 0;
  }

  std::cout << "--- FINISHED TRAINING Exiting ---\n";

  std::vector<std::vector<double>> testInputs =
      read_csv("data/fashion_mnist_test_vectors.csv", 784);
  // std::vector<std::vector<double>> testLabels =
  //     read_csv("data/fashion_mnist_test_labels.csv", 1);

  // Normalize Input
  for (auto &vec : testInputs) {
    for (auto &t : vec) {
      t = t / 255.0;
    }
  }

  std::cout << "\nFINISHED TEST\n";

  // WRITE INTO FILES
  write_output("train_predictions.csv", trainingInputs);
  write_output("test_predictions.csv", testInputs);

  return 0;
}
