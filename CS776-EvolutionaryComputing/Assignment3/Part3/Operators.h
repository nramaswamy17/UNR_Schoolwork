#ifndef OPERATORS_H
#define OPERATORS_H

#include "Tour.h"
#include "Utils.h"
#include <vector>
#include <algorithm>
#include <unordered_map>

class Operators {
public:
    // PMX (Partially Mapped Crossover)
    static std::pair<Tour, Tour> pmxCrossover(const Tour& parent1, const Tour& parent2) {
        int n = parent1.getCities().size();
        
        // Select two crossover points
        int point1 = Utils::randomInt(0, n - 1);
        int point2 = Utils::randomInt(0, n - 1);
        
        if (point1 > point2) {
            std::swap(point1, point2);
        }
        
        // Create offspring
        std::vector<int> offspring1(n, -1);
        std::vector<int> offspring2(n, -1);
        
        // Copy the segment between crossover points
        for (int i = point1; i <= point2; i++) {
            offspring1[i] = parent1.getCities()[i];
            offspring2[i] = parent2.getCities()[i];
        }
        
        // Create mapping for offspring1
        std::unordered_map<int, int> mapping1;
        for (int i = point1; i <= point2; i++) {
            mapping1[parent1.getCities()[i]] = parent2.getCities()[i];
        }
        
        // Create mapping for offspring2
        std::unordered_map<int, int> mapping2;
        for (int i = point1; i <= point2; i++) {
            mapping2[parent2.getCities()[i]] = parent1.getCities()[i];
        }
        
        // Fill remaining positions for offspring1
        for (int i = 0; i < n; i++) {
            if (i >= point1 && i <= point2) continue;
            
            int city = parent2.getCities()[i];
            while (mapping1.find(city) != mapping1.end()) {
                city = mapping1[city];
            }
            offspring1[i] = city;
        }
        
        // Fill remaining positions for offspring2
        for (int i = 0; i < n; i++) {
            if (i >= point1 && i <= point2) continue;
            
            int city = parent1.getCities()[i];
            while (mapping2.find(city) != mapping2.end()) {
                city = mapping2[city];
            }
            offspring2[i] = city;
        }
        
        Tour child1(n);
        Tour child2(n);
        child1.setCities(offspring1);
        child2.setCities(offspring2);
        
        return {child1, child2};
    }
    
    // Swap mutation: swap two random cities
    static void swapMutation(Tour& tour) {
        auto& cities = tour.getCities();
        int n = cities.size();
        
        if (n < 2) return;
        
        int pos1 = Utils::randomInt(0, n - 1);
        int pos2 = Utils::randomInt(0, n - 1);
        
        std::swap(cities[pos1], cities[pos2]);
    }
    
    // Invert mutation: reverse a random subsequence
    static void invertMutation(Tour& tour) {
        auto& cities = tour.getCities();
        int n = cities.size();
        
        if (n < 2) return;
        
        int pos1 = Utils::randomInt(0, n - 1);
        int pos2 = Utils::randomInt(0, n - 1);
        
        if (pos1 > pos2) {
            std::swap(pos1, pos2);
        }
        
        std::reverse(cities.begin() + pos1, cities.begin() + pos2 + 1);
    }
    
    // Apply mutation based on rate
    static void mutate(Tour& tour, double mutationRate, bool useInvert = false) {
        if (Utils::randomBool(mutationRate)) {
            if (useInvert) {
                invertMutation(tour);
            } else {
                swapMutation(tour);
            }
        }
    }
};

#endif