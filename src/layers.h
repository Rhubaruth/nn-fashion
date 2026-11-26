#ifndef LAYER_CLASS
#define LAYER_CLASS

#include <cstddef>
#include <vector>

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------
void printVector(const std::vector<double> &vec);


enum class Activation {
  LeakyReLU, // hidden layers
  Linear,
  LogSigmoid
};

// ---------------------------------------------------------------------------
// Dense Layer
// ---------------------------------------------------------------------------
class DenseLayer {
public:
  DenseLayer(size_t numInputs, size_t numNeurons,
             Activation act = Activation::LeakyReLU, unsigned seed = 42);

  // outputs = f(W * input + b)
  const std::vector<double> &forward(const std::vector<double> &input);

  // input: previous layer
  // return: dE_k/dy_j * activationDerivation(potential)    (pass next)
  //
  // Accumulates dE/dw_ji and dE/db_j (b_j -- bias of j-th neuron)
  const std::vector<double> &backward(const DenseLayer &next,
                                      const std::vector<double> &input);
  const std::vector<double> &backward(const std::vector<double> &dE_dy,
                                      const std::vector<double> &input);

  // Call at the start of every mini-batch.
  void zeroGradients();
  void applyGradient(double lr, unsigned int sample_size);

  void printWeights() const;

  size_t getNumInputs() const { return inputSize; }
  size_t getNumNeurons() const { return neuronCount; }
  const std::vector<double> &getValues() const { return outputs; }

  std::vector<double> &getWeights() { return weights; }
  std::vector<double> &getBiases() { return biases; }
  const std::vector<double> &getGradWeights() const { return gradWeights; }
  const std::vector<double> &getGradBiases() const { return gradBiases; }

  const std::vector<double> &getDZ() const { return dz; }

private:
  double activate(double z) const;
  double activationDerivative(double z) const;

  void accumulateGradient(const std::vector<double> &input);

  static constexpr double kLeakyAlpha = 0.01;

  size_t inputSize;
  size_t neuronCount;
  Activation activation;

  // flat vector for weights
  // weights[j * inputSize + k] = weight from input k to neuron j
  std::vector<double> weights;     // neuronCount * inputSize
  std::vector<double> biases;      // neuronCount
  std::vector<double> gradWeights; // same shape as weights
  std::vector<double> gradBiases;  // same shape as biases

  std::vector<double> potentials; // z_j = \Sum_{i in j<-}(w_ji*y_i)
  std::vector<double> outputs;    // y_j = f(z_j)
  std::vector<double> dz;         // dz = (dE_k/dy_j) * activDeri(potential_j)
};

#endif
