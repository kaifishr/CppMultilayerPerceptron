#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <ostream>
#include <vector>
#include <string>
#include <random>
#include <cstdlib>
#include <ctime>
#include "utils.hpp"

class NeuralNetwork {

    public:
        NeuralNetwork(unsigned int _n_inputs, 
                      unsigned int _n_hidden_1, 
                      unsigned int _n_hidden_2, 
                      unsigned int _n_hidden_3, 
                      unsigned int _n_outputs, 
                      unsigned int _n_epochs,
                      double _learning_rate);

        void run(std::vector<Point> data_train, std::vector<Point> data_valid);
        std::mt19937 gen;
        unsigned int n_epochs;
        unsigned int n_inputs;
        unsigned int n_hidden_1;
        unsigned int n_hidden_2;
        unsigned int n_hidden_3;
        unsigned int n_outputs;
        double learning_rate;

    private:
        std::vector<std::vector<double>> W1;
        std::vector<std::vector<double>> W2;
        std::vector<std::vector<double>> W3;
        std::vector<std::vector<double>> W4;

        std::vector<double> b1;
        std::vector<double> b2;
        std::vector<double> b3;
        std::vector<double> b4;

        std::vector<std::vector<double>> dW1;
        std::vector<std::vector<double>> dW2;
        std::vector<std::vector<double>> dW3;
        std::vector<std::vector<double>> dW4;

        std::vector<double> db1;
        std::vector<double> db2;
        std::vector<double> db3;
        std::vector<double> db4;

        std::vector<double> z1;
        std::vector<double> z2;
        std::vector<double> z3;
        std::vector<double> z4;

        std::vector<double> x;
        std::vector<double> x1;
        std::vector<double> x2;
        std::vector<double> x3;
        std::vector<double> x4;
        std::vector<double> y;

        std::vector<double> delta1;
        std::vector<double> delta2;
        std::vector<double> delta3;
        std::vector<double> delta4;

        void he_initialization(std::vector<std::vector<double>>& weights);
        void feedforward();

        std::vector<double> matmul(const std::vector<std::vector<double>>& W, 
                                   const std::vector<double>& x, 
                                   const std::vector<double>& b);

        std::vector<double> relu(const std::vector<double>& x);
        std::vector<double> sigmoid(const std::vector<double>& x);

        double relu_prime(const double z);
        double sigmoid_prime(const double z);

        void backpropagation();

        void comp_delta_init(std::vector<double>& delta, 
                             const std::vector<double>& z,
                             const std::vector<double>& x,
                             const std::vector<double>& y);

        void comp_delta(const std::vector<std::vector<double>>& W,
                        const std::vector<double>& z,
                        const std::vector<double>& delta_old,
                        std::vector<double>& delta);

        void comp_gradients(std::vector<std::vector<double>>& dW, 
                            std::vector<double>& db, 
                            const std::vector<double>& x,
                            const std::vector<double>& delta);

        void gradient_descent();

        void descent(std::vector<std::vector<double>>& W,
                     std::vector<double>& b,
                     const std::vector<std::vector<double>>& dW,
                     const std::vector<double>& db);

        void comp_stats(const std::vector<Point>& data);
        double comp_loss();
        double comp_accuracy();
        
        void comp_prediction_landscape();
        
        std::vector<Point> comp_grid(const unsigned int n_points_x, 
                                                    const unsigned int n_points_y, 
                                                    const double x_min, 
                                                    const double y_min, 
                                                    const double x_max, 
                                                    const double y_max);

        std::vector<double> comp_prediction(const std::vector<Point>& grid);

        void write_pred_to_file(const std::vector<double> pred, 
                                               const unsigned int n_points_x,
                                               const unsigned int n_points_y);
};

#endif /* NEURAL_NETWORK_H */
