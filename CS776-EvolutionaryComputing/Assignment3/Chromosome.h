#ifndef CHROMOSOME_H
#define CHROMOSOME_H

#include "Room.h"
#include <vector>
#include <limits>

struct Chromosome {
    std::vector<Room> rooms;
    double fitness;
    
    Chromosome(int numRooms) 
        : rooms(numRooms), fitness(std::numeric_limits<double>::max()) {}
    
    bool operator<(const Chromosome& other) const {
        return fitness < other.fitness; // Minimization
    }
};

#endif // CHROMOSOME_H