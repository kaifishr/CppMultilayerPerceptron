#include "random_index_generator.hpp"

RandomIndex::RandomIndex(unsigned int size) : size(size) {
    index.resize(size, 0);
    std::iota(index.begin(), index.end(), 0);
}

unsigned int RandomIndex::get() {
    if (counter < size) {
        return index[counter++];
    } else {
        counter = 0;
        unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(index.begin(), index.end(), std::default_random_engine(seed));
        return index[counter++];
    }
}
