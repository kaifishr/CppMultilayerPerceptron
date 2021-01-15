#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include "data_generator.hpp"

std::ostream& operator<< (std::ostream& out, const Point& point) {
    return out << point.x << " " << point.y << " " << point.label;
}

DataGenerator::DataGenerator() : gen{std::random_device()()} { }

void DataGenerator::write_data() const {
    std::ofstream results;
    std::string file_name = "dataset.dat";
    results.open(file_name);
    if (results.fail()) {
        std::cerr << "Error\n";
    } else {
        for (const auto& point : data) {
            results << point << '\n';
        }
        results.close();
    }
}

void DataGenerator::generate_spirals() {
    std::normal_distribution<double> noise(0.0, 0.3);
    std::uniform_real_distribution<double> spread(0.0, 10.0 * 3.14159);
    std::bernoulli_distribution type(0.5);
    static constexpr double multiplier[]{-1.0, +1.0};

    for (std::vector<Point>::iterator it = data.begin(); it != data.end(); ++it) {
        double r = spread(gen);
        unsigned int label = type(gen);
        it->x = multiplier[label] * (-std::cos(r) * r + noise(gen));
        it->y = multiplier[label] * (+std::sin(r) * r + noise(gen));
        it->label = label;
    }
}

void DataGenerator::generate_chessboard() {
    std::normal_distribution<double> noise(0.0, 0.0);
    const double n_tiles_per_dim = 6.0;
    std::uniform_real_distribution<double> position(0.0, n_tiles_per_dim * 3.14159);
    for (auto &point : data) {
        double x = position(gen);
        double y = position(gen);
        point.x = x + noise(gen);
        point.y = y + noise(gen);
        if ((std::sin(x) > 0.0 && std::sin(y) > 0.0) || (std::sin(x) < 0.0 && std::sin(y) < 0.0)) {
            point.label = 0;
        } else {
            point.label = 1;
        } 
    }
}

void DataGenerator::normalize() {
    auto cmp_x = [] (Point const& lhs, Point const& rhs) {return lhs.x < rhs.x;};
    auto cmp_y = [] (Point const& lhs, Point const& rhs) {return lhs.y < rhs.y;};
    std::pair<std::vector<Point>::iterator, std::vector<Point>::iterator> minmax_x;
    std::pair<std::vector<Point>::iterator, std::vector<Point>::iterator> minmax_y;
    minmax_x = std::minmax_element(data.begin(), data.end(), cmp_x);
    minmax_y = std::minmax_element(data.begin(), data.end(), cmp_y);
    const double x_min = minmax_x.first->x;
    const double x_max = minmax_x.second->x;
    const double y_min = minmax_y.first->y;
    const double y_max = minmax_y.second->y;
    std::vector<Point>::iterator it = data.begin();
    for(; it != data.end(); ++it) {
        it->x = 2.0 * (it->x - x_min) / (x_max - x_min) - 1.0;
        it->y = 2.0 * (it->y - y_min) / (y_max - y_min) - 1.0;
    }
}

std::vector<Point> DataGenerator::get(std::string dataset, unsigned int n_samples) {
    data.resize(n_samples, {0.0, 0.0, 0});
    std::vector<Point>::iterator it;
    if (dataset == "spirals") {
        generate_spirals();
    } else if (dataset == "chessboard") {
        generate_chessboard();
    }
    normalize();
    return data;
}
