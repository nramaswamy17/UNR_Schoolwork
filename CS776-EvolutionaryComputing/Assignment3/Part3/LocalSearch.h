#ifndef LOCAL_SEARCH_H
#define LOCAL_SEARCH_H

#include "Tour.h"
#include "TSPInstance.h"
#include <algorithm>
#include <iostream>

class LocalSearch {
public:
    // 2-opt local search
    // Iteratively improves tour by removing two edges and reconnecting in a different way
    static int twoOpt(Tour& tour, const TSPInstance& instance, int maxIterations = 100000) {
        int improvements = 0;
        int iterations = 0;
        bool improved = true;
        
        int n = tour.getCities().size();
        
        // Make sure initial tour length is calculated
        tour.calculateLength(instance);
        
        while (improved && iterations < maxIterations) {
            improved = false;
            
            // Try all possible edge swaps
            for (int i = 0; i < n - 1; i++) {
                for (int j = i + 2; j < n; j++) {
                    // Skip if j is adjacent to i (wrapping around)
                    if (i == 0 && j == n - 1) continue;
                    
                    // Calculate current distance
                    int city_i = tour.getCities()[i];
                    int city_i_next = tour.getCities()[i + 1];
                    int city_j = tour.getCities()[j];
                    int city_j_next = tour.getCities()[(j + 1) % n];
                    
                    int current_dist = instance.getDistance(city_i, city_i_next) + 
                                      instance.getDistance(city_j, city_j_next);
                    
                    // Calculate distance after swap
                    int new_dist = instance.getDistance(city_i, city_j) + 
                                  instance.getDistance(city_i_next, city_j_next);
                    
                    // If improvement found, do the swap
                    if (new_dist < current_dist) {
                        // Reverse the segment between i+1 and j
                        twoOptSwap(tour, i + 1, j);
                        tour.calculateLength(instance); // Recalculate tour length
                        improvements++;
                        improved = true;
                        // Break to restart search with new tour
                        goto restart_search;
                    }
                }
            }
            restart_search:
            iterations++;
        }
        
        return improvements;
    }
    
private:
    // Perform 2-opt swap by reversing segment between start and end
    static void twoOptSwap(Tour& tour, int start, int end) {
        std::vector<int> cities = tour.getCities(); // Make a copy
        std::reverse(cities.begin() + start, cities.begin() + end + 1);
        tour.setCities(cities); // This invalidates the length cache
    }
};

#endif