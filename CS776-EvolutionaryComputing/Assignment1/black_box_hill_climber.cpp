#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <algorithm>

using namespace std;

// External evaluation function from .o files
extern double eval(int *pj);

// Generate random binary vector (0s and 1s only)
vector<int> generate_random_binary_vector(int num_dimensions) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(0, 1);

    vector<int> vec(num_dimensions);
    for (int i = 0; i < num_dimensions; ++i) {
        vec[i] = dist(gen);
    }
    return vec;
}

// Hill climber for black box functions
pair<vector<double>, vector<int>> black_box_hill_climb(
    vector<int> current,
    int max_iterations = 1000
) {
    random_device rd;
    mt19937 gen(rd());
    
    vector<double> fitness_history;
    fitness_history.reserve(max_iterations);
    
    // Convert vector to array for eval function
    int current_array[100];
    for (int i = 0; i < 100; i++) {
        current_array[i] = current[i];
    }
    double current_fitness = eval(current_array);
    
    for (int i = 0; i < max_iterations; ++i) {
        // Store current fitness
        fitness_history.push_back(current_fitness);
        
        // Check if we found the optimum
        if (current_fitness >= 99.9) {  // Close enough to 100
            cout << "Found optimum at iteration " << i << "!" << endl;
            break;
        }
        
        // Generate neighbor by flipping a random bit
        vector<int> neighbor = current;
        uniform_int_distribution<int> bit_dist(0, 99);  // 0 to 99 for 100-element array
        int flip_index = bit_dist(gen);
        neighbor[flip_index] = 1 - neighbor[flip_index];  // Flip the bit
        
        // Convert neighbor to array for eval function
        int neighbor_array[100];
        for (int j = 0; j < 100; j++) {
            neighbor_array[j] = neighbor[j];
        }
        double neighbor_fitness = eval(neighbor_array);
        
        // Move to neighbor if fitness is better (maximization)
        if (neighbor_fitness > current_fitness) {
            current = neighbor;
            current_fitness = neighbor_fitness;
        }
    }
    
    return make_pair(fitness_history, current);
}

// Export results to CSV
void export_black_box_results(const vector<double>& fitness_history, 
                             const vector<int>& final_vector, 
                             const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }
    
    file << "Iteration,Fitness\n";
    for (size_t i = 0; i < fitness_history.size(); ++i) {
        file << i+1 << "," << fitness_history[i] << "\n";
    }
    file.close();
    cout << "Results exported to " << filename << endl;
}

// Run experiment on black box function
void run_black_box_experiment(const string& label, const string& csv_prefix) {
    cout << "\n" << label << ":" << endl;
    
    int successful_runs = 0;
    vector<double> all_final_fitness;
    
    for (int run = 0; run < 30; run++) {
        cout << "Run " << run + 1 << ": ";
        
        // Generate random starting point
        vector<int> initial_candidate = generate_random_binary_vector(100);
        
        // Run hill climber
        auto result = black_box_hill_climb(initial_candidate, 1000);
        vector<double> fitness_history = result.first;
        vector<int> final_candidate = result.second;
        
        double final_fitness = fitness_history.back();
        all_final_fitness.push_back(final_fitness);
        
        cout << "Final fitness: " << final_fitness;
        
        if (final_fitness >= 99.9) {  // Close to optimum
            successful_runs++;
            cout << " (SUCCESS!)";
        }
        cout << endl;
        
        // Export this run's data
        string csv_filename = csv_prefix + to_string(run) + ".csv";
        export_black_box_results(fitness_history, final_candidate, csv_filename);
    }
    
    // Calculate statistics
    double sum = 0;
    for (double f : all_final_fitness) sum += f;
    double average = sum / all_final_fitness.size();
    
    cout << "\n=== SUMMARY ===" << endl;
    cout << "Success rate: " << successful_runs << "/30 (" 
         << (successful_runs * 100.0 / 30.0) << "%)" << endl;
    cout << "Average final fitness: " << average << endl;
    cout << "Best fitness achieved: " << *max_element(all_final_fitness.begin(), all_final_fitness.end()) << endl;
}

// Test the eval function with different inputs
void test_eval_function() {
    cout << "Testing eval function:" << endl;
    
    // Test all zeros
    int all_zeros[100];
    for (int i = 0; i < 100; i++) all_zeros[i] = 0;
    cout << "All zeros: " << eval(all_zeros) << endl;
    
    // Test all ones
    int all_ones[100];
    for (int i = 0; i < 100; i++) all_ones[i] = 1;
    cout << "All ones: " << eval(all_ones) << endl;
    
    // Test half and half
    int half_half[100];
    for (int i = 0; i < 100; i++) half_half[i] = (i < 50) ? 1 : 0;
    cout << "First 50 ones, rest zeros: " << eval(half_half) << endl;
    
    // Test alternating
    int alternating[100];
    for (int i = 0; i < 100; i++) alternating[i] = i % 2;
    cout << "Alternating 0,1,0,1...: " << eval(alternating) << endl;
}

int main(int argc, char* argv[]) {
    // First test the eval function to understand its behavior
    test_eval_function();
    
    // Determine the CSV prefix based on command line argument
    string csv_prefix = "data/blackbox_run_";  // default
    string label = "BLACK BOX OPTIMIZATION";
    
    if (argc > 1) {
        string version = argv[1];
        if (version == "bb1") {
            csv_prefix = "data/bb1_run_";
            label = "BLACK BOX OPTIMIZATION (BB1)";
        } else if (version == "bb2") {
            csv_prefix = "data/bb2_run_";
            label = "BLACK BOX OPTIMIZATION (BB2)";
        }
    }
    
    // Run the hill climbing experiment
    run_black_box_experiment(label, csv_prefix);
    
    return 0;
}
