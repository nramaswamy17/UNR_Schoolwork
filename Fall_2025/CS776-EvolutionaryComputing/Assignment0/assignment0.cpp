/*
My Implementation Plan
Step 1: Sphere Function Evaluation

Create a function that takes a vector of double values (representing a solution)
Calculate the sphere function: f(x) = Σxi²
Return the fitness score as a double

Step 2: Random Solution Generator

Create a function that generates random candidate solutions
Input: number of dimensions needed
Uses random number generation to create values within the search bounds
Returns: a vector containing the random solution

Step 3: Main Function

Call the generator function to create a random solution
Pass that solution to the sphere function to evaluate it
Print the results to test that everything works

*/

#include <iostream>
#include <vector>
#include <stdexcept>
#include "dejong_functions.h"

#include <random>

// Generates a vector of random doubles within the given range [min_val, max_val]
// num_dimensions: number of elements in the vector
// range: a std::pair<double, double> representing [min_val, max_val]
std::vector<double> generate_random_vector(int num_dimensions, std::pair<double, double> range) {
    if (num_dimensions <= 0) {
        throw std::invalid_argument("Number of dimensions must be positive.");
    }
    double min_val = range.first;
    double max_val = range.second;
    if (min_val > max_val) {
        throw std::invalid_argument("Invalid range: min_val is greater than max_val.");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min_val, max_val);

    std::vector<double> vec(num_dimensions);
    for (int i = 0; i < num_dimensions; ++i) {
        vec[i] = dist(gen);
    }
    return vec;
}



// Basic hill climber
std::vector<double> hill_climb(std::vector<double> current, double step_size = .1, int max_iterations = 1000) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, step_size);

    std::vector<double> fitness_history;
    fitness_history.reserve(max_iterations);

    double current_fitness = sphere(current);

    for (int i = 0; i < max_iterations; ++i) {
        // Store current fitness for current iteration
        fitness_history.push_back(current_fitness);
        
        // Generate neighbor by adding small random changes
        std::vector<double> neighbor = current;
        for (size_t j = 0; j < neighbor.size(); ++j) {
            neighbor[j] += dist(gen);
            neighbor[j] = std::max(-5.12, std::min(5.12, neighbor[j]));
        }

        double neighbor_fitness = sphere(neighbor);
        
        // Move to neighbor if the fitness is better
        if (neighbor_fitness < current_fitness) {
            current = neighbor;
            current_fitness = neighbor_fitness;
        }
        
        //std::cout << fitness_history[i] - fitness_history[i-10];

        // If fitness has not improved, return
        if (i >= 10 && (fitness_history[i-10] - fitness_history[i] < .001)){
            return fitness_history;
        }
    }
    return fitness_history;
}

int main() {
    // Example vector for testing
    std::vector<double> test_vec = generate_random_vector(5, {-5.12, 5.12});
    std::cout << "Initial Candidate: [";
    for (size_t i = 0; i < test_vec.size(); ++i) {
        std::cout << test_vec[i];
        if (i != test_vec.size() - 1) std::cout << ", ";
    }
    // Start hill climber
    std::vector<double> fitness_history = hill_climb(test_vec);

    // Example: Print fitness every 100 iterations to see progress
    std::cout << "\nFitness progress:" << std::endl;
    for (size_t i = 0; i < fitness_history.size(); i += 100) {
        std::cout << "Iteration " << i << ": " << fitness_history[i] << std::endl;
    }

    return 0;
}
