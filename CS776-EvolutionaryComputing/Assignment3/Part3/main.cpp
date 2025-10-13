#include "TSPInstance.h"
#include "GeneticAlgorithm.h"
#include "LocalSearch.h"
#include "Utils.h"
#include <iostream>
#include <map>
#include <string>

struct BenchmarkInfo {
    std::string filename;
    int optimalLength;
};

int main(int argc, char* argv[]) {
    // Benchmark problems with optimal tour lengths
    std::map<std::string, BenchmarkInfo> benchmarks = {
        {"burma14",  {"problems/burma14.tsp",  3323}},
        {"berlin52", {"problems/berlin52.tsp", 7542}},
        {"eil51",    {"problems/eil51.tsp",    426}},
        {"eil76",    {"problems/eil76.tsp",    538}},
        {"lin105",   {"problems/lin105.tsp",   14379}},
        {"lin318",   {"problems/lin318.tsp",   42029}}
    };
    
    // Parse command line arguments
    unsigned int seed = 42;
    std::string benchmarkName = "burma14"; // Default
    
    if (argc > 1) {
        benchmarkName = argv[1];
    }
    if (argc > 2) {
        seed = std::stoi(argv[2]);
    }
    
    // Check if benchmark exists
    if (benchmarks.find(benchmarkName) == benchmarks.end()) {
        std::cerr << "Error: Unknown benchmark '" << benchmarkName << "'" << std::endl;
        std::cerr << "Available benchmarks: ";
        for (const auto& pair : benchmarks) {
            std::cerr << pair.first << " ";
        }
        std::cerr << std::endl;
        return 1;
    }
    
    std::cout << "==================================================" << std::endl;
    std::cout << "TSP Genetic Algorithm with CHC Selection" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Benchmark: " << benchmarkName << std::endl;
    std::cout << "Random seed: " << seed << std::endl;
    std::cout << "==================================================" << std::endl;
    
    // Initialize random number generator
    Utils::setSeed(seed);
    
    // Load TSP instance
    TSPInstance instance;
    BenchmarkInfo info = benchmarks[benchmarkName];
    
    if (!instance.loadFromFile(info.filename)) {
        std::cerr << "Failed to load TSP instance" << std::endl;
        return 1;
    }
    
    instance.setOptimalTourLength(info.optimalLength);
    std::cout << "Optimal tour length: " << info.optimalLength << std::endl;
    std::cout << "==================================================" << std::endl;
    
    // Configure GA parameters based on problem size
    GAParameters params;
    int numCities = instance.getNumCities();
    
    if (numCities <= 20) {
        params.populationSize = 50;
        params.maxGenerations = 500;
    } else if (numCities <= 60) {
        params.populationSize = 100;
        params.maxGenerations = 1000;
    } else if (numCities <= 110) {
        params.populationSize = 150;
        params.maxGenerations = 2000;
    } else {
        params.populationSize = 200;
        params.maxGenerations = 3000;
    }
    
    params.crossoverRate = 0.9;
    params.mutationRate = 0.01; // Low rate, mainly for cataclysm
    params.useInvertMutation = true;
    
    // Run GA
    GeneticAlgorithm ga(&instance, params);
    ga.run();
    
    // Report results
    std::cout << "\n==================================================" << std::endl;
    std::cout << "RESULTS" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    const Tour& bestTour = ga.getBestTour();
    int bestLength = bestTour.getLength();
    int optimalLength = instance.getOptimalTourLength();
    
    double percentFromOptimal = ((double)(bestLength - optimalLength) / optimalLength) * 100.0;
    
    std::cout << "Best tour length: " << bestLength << std::endl;
    std::cout << "Optimal tour length: " << optimalLength << std::endl;
    std::cout << "Percent from optimal: " << std::fixed << std::setprecision(2) 
              << percentFromOptimal << "%" << std::endl;
    std::cout << "Found at evaluation: " << ga.getBestEvaluation() << std::endl;
    std::cout << "==================================================" << std::endl;
    
    // Save results
    ga.saveResults("data");
    
    // ======== SUBPART 2: Apply 2-opt Local Search ========
    std::cout << "\n==================================================" << std::endl;
    std::cout << "APPLYING 2-OPT LOCAL SEARCH (Subpart 2)" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    // Make a copy of the best tour from GA
    Tour improvedTour = bestTour;
    int gaBestLength = bestTour.getLength();
    
    std::cout << "Starting 2-opt from GA best tour: " << gaBestLength << std::endl;
    
    // Apply 2-opt
    int improvements = LocalSearch::twoOpt(improvedTour, instance);
    int twoOptLength = improvedTour.getLength();
    
    // Calculate improvement
    int improvement = gaBestLength - twoOptLength;
    double percentImprovement = ((double)improvement / gaBestLength) * 100.0;
    double twoOptPercentFromOptimal = ((double)(twoOptLength - optimalLength) / optimalLength) * 100.0;
    
    std::cout << "\n2-Opt Results:" << std::endl;
    std::cout << "  GA best tour length: " << gaBestLength << std::endl;
    std::cout << "  2-Opt improved tour: " << twoOptLength << std::endl;
    std::cout << "  Improvement: " << improvement << " (" << std::fixed << std::setprecision(2) 
              << percentImprovement << "%)" << std::endl;
    std::cout << "  Number of 2-opt swaps: " << improvements << std::endl;
    std::cout << "  2-Opt distance from optimal: " << twoOptPercentFromOptimal << "%" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    // Save 2-opt results
    std::string twoOptFile = "data/" + instance.getName() + "_2opt_results.txt";
    std::ofstream outFile(twoOptFile);
    if (outFile.is_open()) {
        outFile << "GA Best Tour Length: " << gaBestLength << "\n";
        outFile << "2-Opt Improved Tour Length: " << twoOptLength << "\n";
        outFile << "Improvement: " << improvement << "\n";
        outFile << "Percent Improvement: " << percentImprovement << "%\n";
        outFile << "Number of 2-Opt Swaps: " << improvements << "\n";
        outFile << "Optimal Tour Length: " << optimalLength << "\n";
        outFile << "GA Percent from Optimal: " << percentFromOptimal << "%\n";
        outFile << "2-Opt Percent from Optimal: " << twoOptPercentFromOptimal << "%\n";
        outFile << "2-Opt Tour: ";
        for (int city : improvedTour.getCities()) {
            outFile << city << " ";
        }
        outFile << "\n";
        outFile.close();
        std::cout << "2-Opt results saved to " << twoOptFile << std::endl;
    }
    
    return 0;
}