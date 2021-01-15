#ifndef DATA_GENERATOR_H
#define DATA_GENERATOR_H

#include <ostream>
#include <vector>
#include <string>
#include <random>
#include <ctime>
#include <algorithm>
#include "utils.hpp"

class DataGenerator {

    public:
        //DataGenerator (unsigned int);
        DataGenerator();
        unsigned int n_samples;
        const unsigned int n_dims = 2;
        const unsigned int n_classes = 2;
        std::vector<Point> get(std::string dataset, unsigned int n_points);
        void write_data() const;
        std::mt19937 gen;

    private:
        std::vector<Point> data;
        void generate_spirals();
        void generate_chessboard();
        void normalize();

};

std::ostream& operator<< (std::ostream& out, const Point& point);

#endif /* DATA_GENERATOR_H */
