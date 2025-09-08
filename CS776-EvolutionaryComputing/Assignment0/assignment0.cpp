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
#include <fstream>
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

double get_fitness(const std::vector<double>& candidate, const std::string& function_name) {
    // Make a copy because the De Jong functions expect non-const reference
    std::vector<double> candidate_copy = candidate;
    if (function_name == "sphere") {
        return sphere(candidate_copy);
    } else if (function_name == "rosenbrock") {
        return rosenbrock(candidate_copy);
    } else if (function_name == "step") {
        return step(candidate_copy);
    } else {
        throw std::invalid_argument("Unknown function name: " + function_name);
    }
}


// Basic hill climber
std::pair<std::vector<double>, std::vector<double>> hill_climb(
    std::vector<double> current,
    const std::string& function_name,
    double step_size = 1,
    int max_iterations = 1000,
    std::pair<double, double> bounds = {-5.12, 5.12}
) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<double> fitness_history;
    fitness_history.reserve(max_iterations);

    double current_fitness = get_fitness(current, function_name);

    for (int i = 0; i < max_iterations; ++i) {

        // step size scaling
        float step_size_adj = step_size / (1 + i % 100); // every 100 iterations make the step size smaller

        // Store current fitness for current iteration
        fitness_history.push_back(current_fitness);
        
        // Generate neighbor by adding small random changes
        std::vector<double> neighbor = current;
        for (size_t j = 0; j < neighbor.size(); ++j) {
            // Create a new normal distribution for each dimension to ensure proper randomization
            std::normal_distribution<double> dist(0.0, step_size_adj);
            neighbor[j] += dist(gen);
            neighbor[j] = std::max(bounds.first, std::min(bounds.second, neighbor[j]));
        }

        double neighbor_fitness = get_fitness(neighbor, function_name);
        
        // Move to neighbor if the fitness is better
        if (neighbor_fitness < current_fitness) {
            current = neighbor;
            current_fitness = neighbor_fitness;
        }
        
        //std::cout << fitness_history[i] - fitness_history[i-10];

        // If fitness has not improved, return
        /*
        if (i >= 10 && (fitness_history[i-10] - fitness_history[i] < .001)){
            return std::make_pair(fitness_history, current);
        }*/
    }
    return std::make_pair(fitness_history, current);
}


void exportFitnessToCSV(const std::vector<double>& fitness_history, const std::vector<double>& final_vector, const std::string& filename = "fitness_data.csv") {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }
    
    // Create header with vector dimensions
    file << "Iteration,Fitness";
    for (size_t i = 0; i < final_vector.size(); ++i) {
        file << ",x" << i;
    }
    file << "\n";
    
    for (size_t i = 0; i < fitness_history.size(); ++i) {
        file << i+1 << "," << fitness_history[i];
        for (size_t j = 0; j < final_vector.size(); ++j) {
            file << "," << final_vector[j];
        }
        file << "\n";
    }
    
    file.close();
    //std::cout << "Fitness data exported to " << filename << "\n";
}

// Reusable experiment runner
void run_experiment(const std::string& label,
                    const std::string& function_name,
                    int dimensions,
                    double range,
                    const std::string& csv_prefix) {
    std::cout << "\n" << label << ":" << std::endl;

    // Helper lambda to print a vector
    auto print_vector = [](const std::vector<double>& vec) {
        std::cout << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            std::cout << vec[i];
            if (i != vec.size() - 1) std::cout << ", ";
        }
        std::cout << "]";
    };

    std::cout << "Run # | Fitness | Initial Candidate | Final Candidate:" << std::endl;
    for (int run = 0; run < 10000; run++) {
        std::vector<double> initial_candidate = generate_random_vector(dimensions, {-range, range});

        // Determine bounds per function
        std::pair<double, double> bounds = {-5.12, 5.12};
        if (function_name == "rosenbrock") bounds = {-2.048, 2.048};

        auto result = hill_climb(initial_candidate, function_name, .1, 1000, bounds); // Start hill climber
        std::vector<double> fitness_history = result.first;
        std::vector<double> final_candidate = result.second;
        /*
        std::cout << run;
        std::cout << " | ";
        std::cout << fitness_history.back();
        std::cout << " | ";
        print_vector(initial_candidate);
        std::cout << "\t-> ";
        print_vector(final_candidate);
        std::cout << std::endl;
        */
        // Export fitness history to CSV, including the run number in the CSV
        std::string csv_filename = csv_prefix + std::to_string(run) + ".csv";
        exportFitnessToCSV(fitness_history, final_candidate, csv_filename);
    }
}

int main() {

    int n; 
    double range;
    std::vector<double> initial_candidate;
    std::vector<double> fitness_history;
    std::vector<double> final_candidate;

    // Helper lambda to print a vector
    auto print_vector = [](const std::vector<double>& vec) {
        std::cout << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            std::cout << vec[i];
            if (i != vec.size() - 1) std::cout << ", ";
        }
        std::cout << "]";
    };
    
    // Sphere
    run_experiment("SPHERE FUNCTION", "sphere", 3, 5.12, "data/sphere_data_run");

    // Rosenbrock
    run_experiment("ROSENBROCK FUNCTION", "rosenbrock", 2, 2.048, "data/rosenbrock_data_run");

    // Step
    run_experiment("STEP FUNCTION", "step", 5, 5.12, "data/step_data_run");
    return 0;
}
