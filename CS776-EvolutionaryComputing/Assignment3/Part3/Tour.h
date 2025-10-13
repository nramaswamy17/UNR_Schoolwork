#ifndef TOUR_H
#define TOUR_H

#include <vector>
#include <algorithm>
#include <numeric>
#include "TSPInstance.h"
#include "Utils.h"

class Tour {
private:
    std::vector<int> cities;
    int tourLength;
    bool lengthValid;
    
public:
    Tour() : tourLength(0), lengthValid(false) {}
    
    Tour(int numCities) : tourLength(0), lengthValid(false) {
        cities.resize(numCities);
        for (int i = 0; i < numCities; i++) {
            cities[i] = i;
        }
    }
    
    void randomize() {
        Utils::shuffleVector(cities);
        lengthValid = false;
    }
    
    int calculateLength(const TSPInstance& instance) {
        if (lengthValid) {
            return tourLength;
        }
        
        tourLength = 0;
        int n = cities.size();
        
        for (int i = 0; i < n; i++) {
            int from = cities[i];
            int to = cities[(i + 1) % n];
            tourLength += instance.getDistance(from, to);
        }
        
        lengthValid = true;
        return tourLength;
    }
    
    int getLength() const { return tourLength; }
    
    double getFitness() const {
        // Fitness for maximization: higher is better
        // We want to minimize tour length, so fitness = 1.0 / tourLength
        if (tourLength == 0) return 0.0;
        return 1.0 / static_cast<double>(tourLength);
    }
    
    const std::vector<int>& getCities() const { return cities; }
    std::vector<int>& getCities() { return cities; }
    
    void setCities(const std::vector<int>& newCities) { 
        cities = newCities; 
        lengthValid = false;
    }
    
    int hammingDistance(const Tour& other) const {
        // Count number of positions where cities differ
        int distance = 0;
        int n = cities.size();
        
        for (int i = 0; i < n; i++) {
            if (cities[i] != other.cities[i]) {
                distance++;
            }
        }
        
        return distance;
    }
    
    bool operator<(const Tour& other) const {
        return tourLength < other.tourLength; // For sorting (minimize)
    }
};

#endif