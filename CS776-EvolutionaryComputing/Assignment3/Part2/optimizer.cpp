#include <iostream>
#include <vector>
#include <cstring>
#include <sys/stat.h>
#include <cmath>
#include "geneticAlgorithm.h"
#include "hillClimber.h"

// Create data directory if it doesn't exist
void createDataDir() {
    struct stat st;
    if (stat("data", &st) == -1) {
        mkdir("data", 0777);
    }
}

int main(int argc, char* argv[]) {
    createDataDir();
    
    if (argc < 2) {
        std::cout << "Usage:\n";
        std::cout << "  GA:  " << argv[0] << " ga <pop_size> <generations> <mut_rate> <cross_rate> <tourn_size> <num_runs>\n";
        std::cout << "  HC:  " << argv[0] << " hc <iterations> <num_runs>\n";
        std::cout << "\nExamples:\n";
        std::cout << "  " << argv[0] << " ga 100 50 0.01 0.9 3 30\n";
        std::cout << "  " << argv[0] << " hc 10000 30\n";
        return 1;
    }
    
    std::string mode = argv[1];
    
    if (mode == "ga") {
        if (argc < 8) {
            std::cerr << "GA requires: pop_size generations mut_rate cross_rate tourn_size num_runs\n";
            return 1;
        }
        
        int pop_size = std::stoi(argv[2]);
        int generations = std::stoi(argv[3]);
        double mut_rate = std::stod(argv[4]);
        double cross_rate = std::stod(argv[5]);
        int tourn_size = std::stoi(argv[6]);
        int num_runs = std::stoi(argv[7]);
        
        std::cout << "Running GA with parameters:\n";
        std::cout << "  Population: " << pop_size << "\n";
        std::cout << "  Generations: " << generations << "\n";
        std::cout << "  Mutation rate: " << mut_rate << "\n";
        std::cout << "  Crossover rate: " << cross_rate << "\n";
        std::cout << "  Tournament size: " << tourn_size << "\n";
        std::cout << "  Number of runs: " << num_runs << "\n\n";
        
        std::vector<double> best_fitnesses;
        
        for (int run = 0; run < num_runs; run++) {
            int seed = 1000 + run;
            GeneticAlgorithm ga(pop_size, generations, mut_rate, cross_rate, tourn_size, seed);
            double best_fitness = ga.run();
            best_fitnesses.push_back(best_fitness);
            
            // Save individual run history
            std::string filename = "data/ga_run_" + std::to_string(run) + ".txt";
            ga.saveHistory(filename);
            
            std::cout << "Run " << (run + 1) << "/" << num_runs 
                     << ": Best fitness = " << best_fitness << std::endl;
        }
        
        // Calculate statistics
        double sum = 0.0, sum_sq = 0.0;
        double max_fit = best_fitnesses[0];
        double min_fit = best_fitnesses[0];
        
        for (double f : best_fitnesses) {
            sum += f;
            sum_sq += f * f;
            if (f > max_fit) max_fit = f;
            if (f < min_fit) min_fit = f;
        }
        
        double mean = sum / num_runs;
        double variance = (sum_sq / num_runs) - (mean * mean);
        double std_dev = sqrt(variance);
        
        // Save summary
        std::ofstream summary("data/ga_summary.txt");
        summary << "# GA Results Summary\n";
        summary << "# Parameters: pop=" << pop_size << " gen=" << generations 
                << " mut=" << mut_rate << " cross=" << cross_rate 
                << " tourn=" << tourn_size << "\n";
        summary << "# Runs: " << num_runs << "\n";
        summary << "Mean: " << mean << "\n";
        summary << "StdDev: " << std_dev << "\n";
        summary << "Max: " << max_fit << "\n";
        summary << "Min: " << min_fit << "\n";
        summary << "\n# All best fitnesses:\n";
        for (size_t i = 0; i < best_fitnesses.size(); i++) {
            summary << "Run " << i << ": " << best_fitnesses[i] << "\n";
        }
        summary.close();
        
        std::cout << "\n=== Summary ===\n";
        std::cout << "Mean fitness: " << mean << " ± " << std_dev << "\n";
        std::cout << "Max fitness: " << max_fit << "\n";
        std::cout << "Min fitness: " << min_fit << "\n";
        std::cout << "\nResults saved to data/\n";
        
    } else if (mode == "hc") {
        if (argc < 4) {
            std::cerr << "HC requires: iterations num_runs\n";
            return 1;
        }
        
        int iterations = std::stoi(argv[2]);
        int num_runs = std::stoi(argv[3]);
        
        std::cout << "Running Hill Climber with parameters:\n";
        std::cout << "  Iterations: " << iterations << "\n";
        std::cout << "  Number of runs: " << num_runs << "\n\n";
        
        std::vector<double> best_fitnesses;
        
        for (int run = 0; run < num_runs; run++) {
            int seed = 1000 + run;
            std::mt19937 rng_temp(seed);
            Individual start;
            start.randomize(rng_temp);
            
            HillClimber hc(iterations, seed);
            double best_fitness = hc.run(start);
            best_fitnesses.push_back(best_fitness);
            
            // Save individual run history
            std::string filename = "data/hc_run_" + std::to_string(run) + ".txt";
            hc.saveHistory(filename);
            
            std::cout << "Run " << (run + 1) << "/" << num_runs 
                     << ": Best fitness = " << best_fitness << std::endl;
        }
        
        // Calculate statistics
        double sum = 0.0, sum_sq = 0.0;
        double max_fit = best_fitnesses[0];
        double min_fit = best_fitnesses[0];
        
        for (double f : best_fitnesses) {
            sum += f;
            sum_sq += f * f;
            if (f > max_fit) max_fit = f;
            if (f < min_fit) min_fit = f;
        }
        
        double mean = sum / num_runs;
        double variance = (sum_sq / num_runs) - (mean * mean);
        double std_dev = sqrt(variance);
        
        // Save summary
        std::ofstream summary("data/hc_summary.txt");
        summary << "# Hill Climber Results Summary\n";
        summary << "# Parameters: iterations=" << iterations << "\n";
        summary << "# Runs: " << num_runs << "\n";
        summary << "Mean: " << mean << "\n";
        summary << "StdDev: " << std_dev << "\n";
        summary << "Max: " << max_fit << "\n";
        summary << "Min: " << min_fit << "\n";
        summary << "\n# All best fitnesses:\n";
        for (size_t i = 0; i < best_fitnesses.size(); i++) {
            summary << "Run " << i << ": " << best_fitnesses[i] << "\n";
        }
        summary.close();
        
        std::cout << "\n=== Summary ===\n";
        std::cout << "Mean fitness: " << mean << " ± " << std_dev << "\n";
        std::cout << "Max fitness: " << max_fit << "\n";
        std::cout << "Min fitness: " << min_fit << "\n";
        std::cout << "\nResults saved to data/\n";
        
    } else {
        std::cerr << "Unknown mode: " << mode << "\n";
        std::cerr << "Use 'ga' or 'hc'\n";
        return 1;
    }
    
    return 0;
}