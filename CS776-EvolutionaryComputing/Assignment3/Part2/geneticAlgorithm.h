#ifndef GENETIC_ALGORITHM_H
#define GENETIC_ALGORITHM_H

#include <vector>
#include <random>
#include <algorithm>
#include <fstream>

// External eval function from object file
extern double eval(int* vec);

const int GENOME_SIZE = 120;

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

#endif // GENETIC_ALGORITHM_H