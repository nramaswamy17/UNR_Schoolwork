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


int main() {
    // Example vector for testing
    std::vector<double> test_vec = generate_random_vector(5, {-5.12, 5.12});
    double result = sphere(test_vec);
    std::cout << "Initial Candidate: [";
    for (size_t i = 0; i < test_vec.size(); ++i) {
        std::cout << test_vec[i];
        if (i != test_vec.size() - 1) std::cout << ", ";
    }
    std::cout << "] | " << result << std::endl;

    return 0;
}
