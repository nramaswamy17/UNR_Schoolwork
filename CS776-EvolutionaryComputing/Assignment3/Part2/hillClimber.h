#ifndef HILL_CLIMBER_H
#define HILL_CLIMBER_H

#include "geneticAlgorithm.h"
#include <vector>
#include <random>
#include <fstream>

class HillClimber {
private:
    int max_iterations;
    std::mt19937 rng;
    std::vector<int> eval_counts;
    std::vector<double> fitness_history;
    
public:
    HillClimber(int max_iter, int seed) : max_iterations(max_iter), rng(seed) {}
    
    double run(Individual start) {
        Individual current = start;
        current.evaluate();
        
        fitness_history.push_back(current.fitness);
        eval_counts.push_back(1);
        
        std::uniform_int_distribution<int> pos_dist(0, GENOME_SIZE - 1);
        
        for (int iter = 1; iter < max_iterations; iter++) {
            // Random neighbor (flip one bit)
            Individual neighbor = current;
            int flip_pos = pos_dist(rng);
            neighbor.genome[flip_pos] = 1 - neighbor.genome[flip_pos];
            neighbor.evaluate();
            
            // Accept if better
            if (neighbor.fitness > current.fitness) {
                current = neighbor;
            }
            
            // Record every 100 iterations
            if (iter % 100 == 0 || iter == max_iterations - 1) {
                fitness_history.push_back(current.fitness);
                eval_counts.push_back(iter + 1);
            }
        }
        
        return current.fitness;
    }
    
    void saveHistory(const std::string& filename) {
        std::ofstream file(filename);
        file << "# Evaluations Fitness\n";
        for (size_t i = 0; i < fitness_history.size(); i++) {
            file << eval_counts[i] << " " << fitness_history[i] << "\n";
        }
        file.close();
    }
};

#endif // HILL_CLIMBER_H