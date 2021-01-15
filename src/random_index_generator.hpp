#ifndef RANDOM_INDEX_H
#define RANDOM_INDEX_H

#include <random>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>

class RandomIndex {
    public:
        RandomIndex(unsigned int size);
        unsigned int get();
    
    private:
        const unsigned int size;
        unsigned int counter = 0;
        std::vector<unsigned int> index;
};

#endif
