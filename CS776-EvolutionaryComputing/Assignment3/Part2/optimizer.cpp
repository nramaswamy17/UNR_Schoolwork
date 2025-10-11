#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <cstring>
#include <sys/stat.h>

// External eval function from object file
extern double eval(int* vec);

const int GENOME_SIZE = 120;

// Create data directory if it doesn't exist
void createDataDir() {
    struct stat st;
    if (stat("data", &st) == -1) {
        mkdir("data", 0777);
    }
}

class Individual {
public:
    std::vector<int> genome;
    double fitness;
    
    Individual() : genome(GENOME_SIZE, 0), fitness(0.0) {}
    
    void evaluate() {
        fitness = eval(genome.data());
    }
    
    void randomize(std::mt19937& rng) {
        std::uniform_int_distribution<int> dist(0, 1);
        for (int i = 0; i < GENOME_SIZE; i++) {
            genome[i] = dist(rng);
        }
    }
};

class GeneticAlgorithm {
private:
    int pop_size;
    int num_generations;
    double mutation_rate;
    double crossover_rate;
    int tournament_size;
    std::mt19937 rng;
    
    std::vector<Individual> population;
    std::vector<int> eval_counts;
    std::vector<double> max_fitness_history;
    std::vector<double> avg_fitness_history;
    
public:
    GeneticAlgorithm(int pop, int gen, double mut, double cross, int tourn, int seed)
        : pop_size(pop), num_generations(gen), mutation_rate(mut), 
          crossover_rate(cross), tournament_size(tourn), rng(seed) {}
    
    // Tournament selection
    Individual& tournamentSelect() {
        std::uniform_int_distribution<int> dist(0, pop_size - 1);
        int best_idx = dist(rng);
        double best_fitness = population[best_idx].fitness;
        
        for (int i = 1; i < tournament_size; i++) {
            int idx = dist(rng);
            if (population[idx].fitness > best_fitness) {
                best_fitness = population[idx].fitness;
                best_idx = idx;
            }
        }
        return population[best_idx];
    }
    
    // Uniform crossover
    Individual uniformCrossover(Individual& p1, Individual& p2) {
        Individual child;
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        for (int i = 0; i < GENOME_SIZE; i++) {
            child.genome[i] = (dist(rng) < 0.5) ? p1.genome[i] : p2.genome[i];
        }
        return child;
    }
    
    // Bit flip mutation
    void mutate(Individual& ind) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < GENOME_SIZE; i++) {
            if (dist(rng) < mutation_rate) {
                ind.genome[i] = 1 - ind.genome[i];
            }
        }
    }
    
    double run() {
        // Initialize population
        population.resize(pop_size);
        for (auto& ind : population) {
            ind.randomize(rng);
            ind.evaluate();
        }
        
        int eval_count = pop_size;
        
        // Evolution loop
        for (int gen = 0; gen < num_generations; gen++) {
            // Calculate statistics
            double max_fit = population[0].fitness;
            double sum_fit = 0.0;
            for (auto& ind : population) {
                if (ind.fitness > max_fit) max_fit = ind.fitness;
                sum_fit += ind.fitness;
            }
            
            eval_counts.push_back(eval_count);
            max_fitness_history.push_back(max_fit);
            avg_fitness_history.push_back(sum_fit / pop_size);
            
            // Create new population
            std::vector<Individual> new_pop;
            
            // Elitism: keep best individual
            auto best_it = std::max_element(population.begin(), population.end(),
                [](const Individual& a, const Individual& b) { return a.fitness < b.fitness; });
            new_pop.push_back(*best_it);
            
            // Generate offspring
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            while (new_pop.size() < static_cast<size_t>(pop_size)) {
                Individual& p1 = tournamentSelect();
                Individual& p2 = tournamentSelect();
                
                Individual child;
                if (dist(rng) < crossover_rate) {
                    child = uniformCrossover(p1, p2);
                } else {
                    child = p1;
                }
                
                mutate(child);
                child.evaluate();
                eval_count++;
                
                new_pop.push_back(child);
            }
            
            population = new_pop;
        }
        
        // Final statistics
        double max_fit = population[0].fitness;
        double sum_fit = 0.0;
        for (auto& ind : population) {
            if (ind.fitness > max_fit) max_fit = ind.fitness;
            sum_fit += ind.fitness;
        }
        eval_counts.push_back(eval_count);
        max_fitness_history.push_back(max_fit);
        avg_fitness_history.push_back(sum_fit / pop_size);
        
        return max_fit;
    }
    
    void saveHistory(const std::string& filename) {
        std::ofstream file(filename);
        file << "# Evaluations MaxFitness AvgFitness\n";
        for (size_t i = 0; i < max_fitness_history.size(); i++) {
            file << eval_counts[i] << " " 
                 << max_fitness_history[i] << " " 
                 << avg_fitness_history[i] << "\n";
        }
        file.close();
    }
    
    Individual getBest() {
        auto best_it = std::max_element(population.begin(), population.end(),
            [](const Individual& a, const Individual& b) { return a.fitness < b.fitness; });
        return *best_it;
    }
};

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