#include "layers.h"

#include <iostream>
#include <math.h>

#include <random>

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------
void printVector(const std::vector<double> &vec) {
  for (double v : vec) {
    std::cout << std::round(v * 1000.0) / 1000.0 << " ";
  }
  std::cout << "\n";
}

double logistic_sigmoid(double x) { return 1 / (1 + exp(-x)); }


// ---------------------------------------------------------------------------
// DenseLayer
// ---------------------------------------------------------------------------
DenseLayer::DenseLayer(size_t numInputs, size_t numNeurons, Activation act,
                       unsigned seed)
    : inputSize(numInputs), neuronCount(numNeurons), activation(act),
      weights(numNeurons * numInputs), biases(numNeurons, 0),
      gradWeights(numNeurons * numInputs, 0.0), gradBiases(numNeurons, 0.0),
      potentials(numNeurons, 0.0), outputs(numNeurons, 0.0),
      dz(numNeurons, 0.0) {

  // std::srand(seed);
  std::mt19937 gen(seed);
  std::normal_distribution<double> dist(0.0, 2.0/numInputs);
  for (double &w : weights)
    w = dist(gen);
}

void DenseLayer::printWeights() const {
  for (size_t j = 0; j < neuronCount; j++) {
    std::cout << biases[j] << " | ";
    for (size_t k = 0; k < inputSize; k++) {
      std::cout << std::round(weights[j * inputSize + k] * 1000.0) / 1000.0
                << " ";
    }
    std::cout << "\n";
  }
}

const std::vector<double> &
DenseLayer::forward(const std::vector<double> &input) {
  if (input.size() != inputSize) {
    throw std::invalid_argument("DenseLayer::forward: expected input size " +
                                std::to_string(inputSize) + ", got " +
                                std::to_string(input.size()));
  }

  for (size_t j = 0; j < neuronCount; j++) {
    double sum = biases[j];

    for (size_t k = 0; k < inputSize; k++) {
      sum += weights[j * inputSize + k] * input[k];
    }

    potentials[j] = sum;
    outputs[j] = activate(sum);
  }

  return outputs;
}

void DenseLayer::accumulateGradient(const std::vector<double> &input) {
  // accumulate gradW_ji [dE_k/dwji] = dOut_j * activDeri(xi_j) * y_i
  // accumulate gradB_j  [dE_k/dbj]  = dOut_j * activDeri(xi_j) * 1

  for (size_t j = 0; j < neuronCount; j++) {
    gradBiases[j] += dz[j];
    for (size_t i = 0; i < inputSize; i++) {
      gradWeights[j * inputSize + i] += dz[j] * input[i];
    }
  }
}

const std::vector<double> &
DenseLayer::backward(const std::vector<double> &dE_dy,
                     const std::vector<double> &input) {
  // if LastLayer
  // dOut [dE_k/dy_j] = y_j - d_kj [based on loss]

  for (size_t j = 0; j < neuronCount; j++) {
    dz[j] = dE_dy[j] * activationDerivative(potentials[j]);
  }

  accumulateGradient(input);

  return dz;
}

const std::vector<double> &
DenseLayer::backward(const DenseLayer &next, const std::vector<double> &input) {
  // if not LastLayer
  // dOut [dE_k/dy_j] = Sum_{r in j->}(dOut_r*activDeri_r(xi_r)*w_rj)
  std::fill(dz.begin(), dz.end(), 0.0);

  for (size_t r = 0; r < next.getNumNeurons(); r++) {
    double dE_dy = 0.0;
    for (size_t j = 0; j < neuronCount; j++) {
      dE_dy = next.dz[r] * next.weights[r * next.inputSize + j];
      dz[j] += dE_dy;
    }
  }

  for (size_t j = 0; j < neuronCount; j++) {
    dz[j] = dz[j] * activationDerivative(potentials[j]);
  }

  accumulateGradient(input);

  return dz;
}

void DenseLayer::zeroGradients() {
  std::fill(gradWeights.begin(), gradWeights.end(), 0.0);
  std::fill(gradBiases.begin(), gradBiases.end(), 0.0);
}

void DenseLayer::applyGradient(double lr, unsigned int sample_size) {
  double scale = lr / double(sample_size);
  for (size_t i = 0; i < weights.size(); i++) {
    weights[i] -= gradWeights[i] * scale;
  }
  for (size_t i = 0; i < biases.size(); i++) {
    biases[i] -= gradBiases[i] * scale;
  }
  zeroGradients();
}

double DenseLayer::activate(double z) const {
  switch (activation) {
  case Activation::LeakyReLU:
    return z > 0.0 ? z : kLeakyAlpha * z;
  case Activation::Linear:
    return z;
  case Activation::LogSigmoid:
    return logistic_sigmoid(z);
  }
  return z;
}

double DenseLayer::activationDerivative(double z) const {
  switch (activation) {
  case Activation::LeakyReLU:
    return z > 0.0 ? 1.0 : kLeakyAlpha;
  case Activation::Linear:
    return 1.0;
  case Activation::LogSigmoid:
    double s = logistic_sigmoid(z);
    return s * (1 - s);
  }
  return 1.0;
}
