/*
Adapted Hill Climber for Binary Optimization
Supports both continuous DeJong functions and binary functions (OneMax, Deceptive Trap)
*/

#include <iostream>
#include <vector>
#include <stdexcept>
#include <fstream>
#include "dejong_functions.h"
#include <random>
#include <algorithm>

// ========== BINARY FUNCTION IMPLEMENTATIONS ==========

// Easy Function: OneMax - counts the number of 1s
double onemax_eval(const std::vector<int>& solution) {
    int count = 0;
    for (int i = 0; i < solution.size(); i++) {
        count += solution[i];
    }
    return (double)count;
}

// Hard Function: Deceptive Trap Function
double deceptive_trap_eval(const std::vector<int>& solution) {
    double total = 0.0;
    int block_size = 10;
    int num_blocks = solution.size() / block_size;
    
    // Process each block of 10 bits
    for (int block = 0; block < num_blocks; block++) {
        int ones_count = 0;
        for (int i = 0; i < block_size; i++) {
            ones_count += solution[block * block_size + i];
        }
        
        if (ones_count == block_size) {
            total += 10.0;  // Complete block: maximum reward
        } else {
            total += (9.0 - ones_count);  // Incomplete block: deceptive fitness
        }
    }
    return total;
}

// ========== RANDOM GENERATION FUNCTIONS ==========

// Generate random binary vector (0s and 1s)
std::vector<int> generate_random_binary_vector(int num_dimensions) {
    if (num_dimensions <= 0) {
        throw std::invalid_argument("Number of dimensions must be positive.");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 1);

    std::vector<int> vec(num_dimensions);
    for (int i = 0; i < num_dimensions; ++i) {
        vec[i] = dist(gen);
    }
    return vec;
}

// Original function for continuous values (DeJong functions)
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

// ========== FITNESS EVALUATION FUNCTIONS ==========

// For continuous DeJong functions
double get_fitness(const std::vector<double>& candidate, const std::string& function_name) {
    std::vector<double> candidate_copy = candidate;
    if (function_name == "sphere") {
        return sphere(candidate_copy);
    } else if (function_name == "rosenbrock") {
        return rosenbrock(candidate_copy);
    } else if (function_name == "step") {
        return step(candidate_copy);
    } else {
        throw std::invalid_argument("Unknown continuous function name: " + function_name);
    }
}

// For binary functions
double get_binary_fitness(const std::vector<int>& candidate, const std::string& function_name) {
    if (function_name == "onemax") {
        return onemax_eval(candidate);
    } else if (function_name == "deceptive_trap") {
        return deceptive_trap_eval(candidate);
    } else {
        throw std::invalid_argument("Unknown binary function name: " + function_name);
    }
}

// ========== HILL CLIMBING ALGORITHMS ==========

// Binary hill climber using bit-flip mutations
std::pair<std::vector<double>, std::vector<int>> binary_hill_climb(
    std::vector<int> current,
    const std::string& function_name,
    int max_iterations = 1000
) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::vector<double> fitness_history;
    fitness_history.reserve(max_iterations);
    
    double current_fitness = get_binary_fitness(current, function_name);
    
    for (int i = 0; i < max_iterations; ++i) {
        // Store current fitness
        fitness_history.push_back(current_fitness);
        
        // Generate neighbor by flipping a random bit
        std::vector<int> neighbor = current;
        std::uniform_int_distribution<int> bit_dist(0, neighbor.size() - 1);
        int flip_index = bit_dist(gen);
        neighbor[flip_index] = 1 - neighbor[flip_index];  // Flip the bit
        
        double neighbor_fitness = get_binary_fitness(neighbor, function_name);
        
        // Move to neighbor if fitness is better (maximization for binary functions)
        if (neighbor_fitness > current_fitness) {
            current = neighbor;
            current_fitness = neighbor_fitness;
        }
    }
    
    return std::make_pair(fitness_history, current);
}

// Original continuous hill climber
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
        float step_size_adj = step_size / (1 + i % 100);
        fitness_history.push_back(current_fitness);
        
        std::vector<double> neighbor = current;
        for (size_t j = 0; j < neighbor.size(); ++j) {
            std::normal_distribution<double> dist(0.0, step_size_adj);
            neighbor[j] += dist(gen);
            neighbor[j] = std::max(bounds.first, std::min(bounds.second, neighbor[j]));
        }

        double neighbor_fitness = get_fitness(neighbor, function_name);
        
        if (neighbor_fitness < current_fitness) {  // Minimization for DeJong functions
            current = neighbor;
            current_fitness = neighbor_fitness;
        }
    }
    return std::make_pair(fitness_history, current);
}

// ========== CSV EXPORT FUNCTIONS ==========

// For continuous functions
void exportFitnessToCSV(const std::vector<double>& fitness_history, const std::vector<double>& final_vector, const std::string& filename = "fitness_data.csv") {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }
    
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
}

// For binary functions
void exportBinaryFitnessToCSV(const std::vector<double>& fitness_history, const std::vector<int>& final_vector, const std::string& filename = "fitness_data.csv") {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }
    
    file << "Iteration,Fitness";
    for (size_t i = 0; i < final_vector.size(); ++i) {
        file << ",bit" << i;
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
}

// ========== EXPERIMENT RUNNERS ==========

// For continuous functions (your original)
void run_experiment(const std::string& label,
                    const std::string& function_name,
                    int dimensions,
                    double range,
                    const std::string& csv_prefix) {
    std::cout << "\n" << label << ":" << std::endl;
    
    auto print_vector = [](const std::vector<double>& vec) {
        std::cout << "[";
        for (size_t i = 0; i < std::min((size_t)5, vec.size()); ++i) {
            std::cout << vec[i];
            if (i != std::min((size_t)5, vec.size()) - 1) std::cout << ", ";
        }
        if (vec.size() > 5) std::cout << ", ...";
        std::cout << "]";
    };

    for (int run = 0; run < 30; run++) {  // Reduced to 30 runs as mentioned in assignment
        std::vector<double> initial_candidate = generate_random_vector(dimensions, {-range, range});

        std::pair<double, double> bounds = {-5.12, 5.12};
        if (function_name == "rosenbrock") bounds = {-2.048, 2.048};

        auto result = hill_climb(initial_candidate, function_name, .1, 1000, bounds);
        std::vector<double> fitness_history = result.first;
        std::vector<double> final_candidate = result.second;

        std::string csv_filename = csv_prefix + std::to_string(run) + ".csv";
        exportFitnessToCSV(fitness_history, final_candidate, csv_filename);
        
        std::cout << "Run " << run << " - Final fitness: " << fitness_history.back() << std::endl;
    }
}

// For binary functions
void run_binary_experiment(const std::string& label,
                          const std::string& function_name,
                          int dimensions,
                          const std::string& csv_prefix) {
    std::cout << "\n" << label << ":" << std::endl;
    
    auto print_binary_vector = [](const std::vector<int>& vec) {
        std::cout << "[";
        for (size_t i = 0; i < std::min((size_t)10, vec.size()); ++i) {
            std::cout << vec[i];
            if (i != std::min((size_t)10, vec.size()) - 1) std::cout << ", ";
        }
        if (vec.size() > 10) std::cout << ", ...";
        std::cout << "]";
    };

    int successful_runs = 0;
    double max_possible = (function_name == "onemax") ? dimensions : dimensions; // 100 for both
    
    for (int run = 0; run < 30; run++) {
        std::vector<int> initial_candidate = generate_random_binary_vector(dimensions);

        auto result = binary_hill_climb(initial_candidate, function_name, 1000);
        std::vector<double> fitness_history = result.first;
        std::vector<int> final_candidate = result.second;

        std::string csv_filename = csv_prefix + std::to_string(run) + ".csv";
        exportBinaryFitnessToCSV(fitness_history, final_candidate, csv_filename);
        
        double final_fitness = fitness_history.back();
        std::cout << "Run " << run << " - Final fitness: " << final_fitness;
        
        if (function_name == "onemax" && final_fitness == max_possible) {
            successful_runs++;
            std::cout << " (OPTIMAL!)";
        } else if (function_name == "deceptive_trap" && final_fitness == 100.0) {
            successful_runs++;
            std::cout << " (OPTIMAL!)";
        }
        std::cout << std::endl;
    }
    
    std::cout << "Success rate: " << successful_runs << "/30 (" 
              << (successful_runs * 100.0 / 30.0) << "%)" << std::endl;
}

// ========== MAIN FUNCTION ==========

int main() {
    /*
    // Test original DeJong functions
    run_experiment("SPHERE FUNCTION", "sphere", 3, 5.12, "data/sphere_data_run");
    run_experiment("ROSENBROCK FUNCTION", "rosenbrock", 2, 2.048, "data/rosenbrock_data_run");
    run_experiment("STEP FUNCTION", "step", 5, 5.12, "data/step_data_run");
    */
    // Test new binary functions
    run_binary_experiment("ONEMAX FUNCTION (Easy Binary)", "onemax", 100, "data/onemax_data_run");
    run_binary_experiment("DECEPTIVE TRAP FUNCTION (Hard Binary)", "deceptive_trap", 100, "data/deceptive_trap_data_run");
    
    return 0;
}