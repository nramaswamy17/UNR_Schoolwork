#ifndef GENETIC_ALGORITHM_H
#define GENETIC_ALGORITHM_H

#include "TSPInstance.h"
#include "Tour.h"
#include "Operators.h"
#include "Utils.h"
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>

struct GAParameters {
    int populationSize;
    int maxGenerations;
    double crossoverRate;
    double mutationRate;
    int initialThreshold; // For CHC
    bool useInvertMutation;
    
    GAParameters() 
        : populationSize(100),
          maxGenerations(5000),
          crossoverRate(0.6),
          mutationRate(0.01),
          initialThreshold(-1), // Will be computed
          useInvertMutation(false) {}
};

struct GenerationStats {
    int generation;
    int bestLength;
    double avgLength;
    double bestFitness;
    double avgFitness;
    int evaluations;
};

class GeneticAlgorithm {
private:
    TSPInstance* instance;
    GAParameters params;
    
    std::vector<Tour> population;
    std::vector<Tour> offspring;
    
    Tour bestTour;
    int bestEvaluation;
    
    std::vector<GenerationStats> statistics;
    
    int threshold; // CHC threshold for incest prevention
    int totalEvaluations;
    
    void initializePopulation() {
        population.clear();
        
        for (int i = 0; i < params.populationSize; i++) {
            Tour tour(instance->getNumCities());
            tour.randomize();
            population.push_back(tour);
        }
        
        evaluatePopulation();
    }
    
    void evaluatePopulation() {
        for (auto& tour : population) {
            tour.calculateLength(*instance);
            totalEvaluations++;
        }
        
        // Find best tour
        auto best = std::min_element(population.begin(), population.end());
        if (bestTour.getLength() == 0 || best->getLength() < bestTour.getLength()) {
            bestTour = *best;
            bestEvaluation = totalEvaluations;
        }
    }
    
    void chcSelection() {
        offspring.clear();
        
        // Shuffle population for random pairing
        std::vector<Tour> parents = population;
        Utils::shuffleVector(parents);
        
        // Mate pairs if they differ by at least threshold positions
        for (size_t i = 0; i + 1 < parents.size(); i += 2) {
            int difference = parents[i].hammingDistance(parents[i + 1]);
            
            if (difference >= threshold && Utils::randomBool(params.crossoverRate)) {
                auto children = Operators::pmxCrossover(parents[i], parents[i + 1]);
                
                children.first.calculateLength(*instance);
                children.second.calculateLength(*instance);
                totalEvaluations += 2;
                
                offspring.push_back(children.first);
                offspring.push_back(children.second);
            }
        }
        
        // Combine parents and offspring, select best for next generation
        std::vector<Tour> combined = population;
        combined.insert(combined.end(), offspring.begin(), offspring.end());
        
        std::sort(combined.begin(), combined.end());
        
        population.clear();
        for (int i = 0; i < params.populationSize && i < (int)combined.size(); i++) {
            population.push_back(combined[i]);
        }
        
        // Update best
        if (population[0].getLength() < bestTour.getLength()) {
            bestTour = population[0];
            bestEvaluation = totalEvaluations;
        }
        
        // Check for convergence
        if (offspring.empty() || offspring.size() < params.populationSize / 10) {
            threshold--;
            
            if (threshold < 0) {
                cataclysm();
                threshold = instance->getNumCities() / 4;
            }
        }
    }
    
    void cataclysm() {
        std::cout << "Cataclysm at generation with threshold reset!" << std::endl;
        
        // Keep only the best tour
        Tour elite = population[0];
        
        // Reinitialize rest of population with mutations of elite
        population.clear();
        population.push_back(elite);
        
        for (int i = 1; i < params.populationSize; i++) {
            Tour mutant = elite;
            
            // Apply multiple mutations
            int numMutations = Utils::randomInt(instance->getNumCities() / 4, 
                                               instance->getNumCities() / 2);
            for (int j = 0; j < numMutations; j++) {
                if (params.useInvertMutation && Utils::randomBool(0.5)) {
                    Operators::invertMutation(mutant);
                } else {
                    Operators::swapMutation(mutant);
                }
            }
            
            mutant.calculateLength(*instance);
            totalEvaluations++;
            population.push_back(mutant);
        }
    }
    
    GenerationStats computeStats(int generation) {
        GenerationStats stats;
        stats.generation = generation;
        stats.evaluations = totalEvaluations;
        
        int sumLength = 0;
        double sumFitness = 0.0;
        
        stats.bestLength = population[0].getLength();
        stats.bestFitness = population[0].getFitness();
        
        for (const auto& tour : population) {
            sumLength += tour.getLength();
            sumFitness += tour.getFitness();
        }
        
        stats.avgLength = static_cast<double>(sumLength) / population.size();
        stats.avgFitness = sumFitness / population.size();
        
        return stats;
    }
    
public:
    GeneticAlgorithm(TSPInstance* inst, const GAParameters& p)
        : instance(inst), params(p), totalEvaluations(0), bestEvaluation(0) {
        
        // Initialize CHC threshold (typically L/4 where L is chromosome length)
        if (params.initialThreshold < 0) {
            threshold = instance->getNumCities() / 2;
        } else {
            threshold = params.initialThreshold;
        }
        
        bestTour = Tour(instance->getNumCities());
    }
    
    void run() {
        std::cout << "Starting GA with CHC selection..." << std::endl;
        std::cout << "Population size: " << params.populationSize << std::endl;
        std::cout << "Max generations: " << params.maxGenerations << std::endl;
        std::cout << "Initial threshold: " << threshold << std::endl;
        
        initializePopulation();
        
        statistics.clear();
        statistics.push_back(computeStats(0));
        
        for (int gen = 1; gen <= params.maxGenerations; gen++) {
            chcSelection();
            
            GenerationStats stats = computeStats(gen);
            statistics.push_back(stats);
            
            if (gen % 50 == 0 || gen == params.maxGenerations) {
                std::cout << "Gen " << std::setw(4) << gen 
                         << " | Best: " << std::setw(6) << stats.bestLength
                         << " | Avg: " << std::setw(8) << std::fixed << std::setprecision(2) << stats.avgLength
                         << " | Evals: " << totalEvaluations
                         << " | Threshold: " << threshold << std::endl;
            }
        }
        
        std::cout << "\nFinal best tour length: " << bestTour.getLength() << std::endl;
        std::cout << "Total evaluations: " << totalEvaluations << std::endl;
    }
    
    void saveResults(const std::string& outputDir) {
        Utils::createDirectory(outputDir);
        
        std::string filename = outputDir + "/" + instance->getName() + "_stats.csv";
        std::ofstream file(filename);
        
        if (!file.is_open()) {
            std::cerr << "Error: Cannot create output file " << filename << std::endl;
            return;
        }
        
        // Write header
        file << "Generation,BestLength,AvgLength,BestFitness,AvgFitness,Evaluations\n";
        
        // Write data
        for (const auto& stats : statistics) {
            file << stats.generation << ","
                 << stats.bestLength << ","
                 << std::fixed << std::setprecision(2) << stats.avgLength << ","
                 << std::setprecision(6) << stats.bestFitness << ","
                 << stats.avgFitness << ","
                 << stats.evaluations << "\n";
        }
        
        file.close();
        std::cout << "Statistics saved to " << filename << std::endl;
        
        // Save best tour
        filename = outputDir + "/" + instance->getName() + "_best_tour.txt";
        file.open(filename);
        
        if (file.is_open()) {
            file << "Tour length: " << bestTour.getLength() << "\n";
            file << "Found at evaluation: " << bestEvaluation << "\n";
            file << "Cities: ";
            for (int city : bestTour.getCities()) {
                file << city << " ";
            }
            file << "\n";
            file.close();
            std::cout << "Best tour saved to " << filename << std::endl;
        }
    }
    
    const Tour& getBestTour() const { return bestTour; }
    int getBestEvaluation() const { return bestEvaluation; }
    const std::vector<GenerationStats>& getStatistics() const { return statistics; }
};

#endif