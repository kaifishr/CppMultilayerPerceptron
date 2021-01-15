#include <vector>
#include <string>
#include <iostream>
#include "data_generator.hpp"
#include "neural_network.hpp"

int main() {

    const std::string dataset = "chessboard"; // spirals, chessboard
    const unsigned int n_samples_train = 20000;
    const unsigned int n_samples_valid = 2000;

    DataGenerator data_generator;

    std::vector<Point> data_train, data_valid;
    data_train = data_generator.get(dataset, n_samples_train);
    data_valid = data_generator.get(dataset, n_samples_valid);
    data_generator.write_data();

    unsigned int n_inputs = 2;
    unsigned int n_hidden_1 = 64;
    unsigned int n_hidden_2 = 64;
    unsigned int n_hidden_3 = 64;
    unsigned int n_outputs = 2;

    unsigned int n_epochs = 1000;
    double learning_rate = 1e-4;

    NeuralNetwork neural_network(n_inputs, n_hidden_1, n_hidden_2, n_hidden_3, n_outputs, n_epochs, learning_rate);
    neural_network.run(data_train, data_valid);

    return 0;
}
