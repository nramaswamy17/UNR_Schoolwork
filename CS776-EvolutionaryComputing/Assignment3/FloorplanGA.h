#ifndef FLOORPLANGA_H
#define FLOORPLANGA_H

#include "RoomSpec.h"
#include "Chromosome.h"
#include "FileExporter.h"
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

class FloorplanGA {
private:
    std::vector<RoomSpec> roomSpecs;
    std::vector<Chromosome> population;
    
    // GA Parameters
    int populationSize;
    int numGenerations;
    double crossoverRate;
    double mutationRate;
    int eliteCount;
    int tournamentSize;
    
    // Random number generation
    std::mt19937 gen;
    
    // Statistics
    std::vector<double> bestFitnessHistory;
    std::vector<double> avgFitnessHistory;
    Chromosome bestSolution;
    
    void initializeRoomSpecs() {
        roomSpecs.push_back(RoomSpec("Living", 8, 20, 8, 20, 120, 300, 1.5, 1.0));
        roomSpecs.push_back(RoomSpec("Kitchen", 6, 18, 6, 18, 50, 120, 0, 2.0));
        roomSpecs.push_back(RoomSpec("Bath", 5.5, 5.5, 8.5, 8.5, 46.75, 46.75, 0, 2.0));
        roomSpecs.push_back(RoomSpec("Hall", 3.5, 5.5, 3.5, 6, 19, 72, 0, 1.0));
        roomSpecs.push_back(RoomSpec("Bed1", 10, 17, 10, 17, 100, 180, 1.5, 1.0));
        roomSpecs.push_back(RoomSpec("Bed2", 9, 20, 9, 20, 100, 180, 1.5, 1.0));
        roomSpecs.push_back(RoomSpec("Bed3", 8, 18, 8, 18, 100, 180, 1.5, 1.0));
    }
    
    Chromosome createRandomChromosome() {
        Chromosome chrome(roomSpecs.size());
        
        for (size_t i = 0; i < roomSpecs.size(); i++) {
            const RoomSpec& spec = roomSpecs[i];
            double length, width;
            
            if (spec.aspectRatio > 0) {
                double minDim = sqrt(spec.minArea / spec.aspectRatio);
                double maxDim = sqrt(spec.maxArea / spec.aspectRatio);
                std::uniform_real_distribution<> dimDist(minDim, maxDim);
                width = dimDist(gen);
                length = width * spec.aspectRatio;
            } else {
                std::uniform_real_distribution<> lengthDist(spec.minLength, spec.maxLength);
                std::uniform_real_distribution<> widthDist(spec.minWidth, spec.maxWidth);
                length = lengthDist(gen);
                width = widthDist(gen);
            }
            
            length = std::max(spec.minLength, std::min(spec.maxLength, length));
            width = std::max(spec.minWidth, std::min(spec.maxWidth, width));
            
            std::uniform_real_distribution<> posDist(0, 50);
            chrome.rooms[i] = Room(length, width, posDist(gen), posDist(gen));
        }
        
        return chrome;
    }
    
    double calculateOverlap(const Room& r1, const Room& r2) {
        double overlapX = std::max(0.0, std::min(r1.x + r1.width, r2.x + r2.width) - std::max(r1.x, r2.x));
        double overlapY = std::max(0.0, std::min(r1.y + r1.length, r2.y + r2.length) - std::max(r1.y, r2.y));
        return overlapX * overlapY;
    }
    
    double calculateFitness(Chromosome& chrome) {
        double cost = 0;
        double penalty = 0;
        
        for (size_t i = 0; i < chrome.rooms.size(); i++) {
            const Room& room = chrome.rooms[i];
            const RoomSpec& spec = roomSpecs[i];
            double area = room.area();
            
            cost += area * spec.costMultiplier;
            
            if (room.length < spec.minLength || room.length > spec.maxLength) {
                penalty += 1000;
            }
            if (room.width < spec.minWidth || room.width > spec.maxWidth) {
                penalty += 1000;
            }
            
            if (area < spec.minArea || area > spec.maxArea) {
                penalty += 500;
            }
            
            if (spec.aspectRatio > 0) {
                double actualRatio = room.length / room.width;
                double ratioDiff = std::abs(actualRatio - spec.aspectRatio);
                if (ratioDiff > 0.1) {
                    penalty += 300 * ratioDiff;
                }
            }
            
            if (room.x < 0 || room.y < 0) {
                penalty += 500;
            }
        }
        
        for (size_t i = 0; i < chrome.rooms.size(); i++) {
            for (size_t j = i + 1; j < chrome.rooms.size(); j++) {
                double overlap = calculateOverlap(chrome.rooms[i], chrome.rooms[j]);
                if (overlap > 0) {
                    penalty += 2000 * overlap;
                }
            }
        }

        // Calculate bounding box and penalize wasted space
        double min_x = std::numeric_limits<double>::max();
        double min_y = std::numeric_limits<double>::max();
        double max_x = 0;
        double max_y = 0;
        double total_room_area = 0;
        
        for (const Room& room : chrome.rooms) {
            min_x = std::min(min_x, room.x);
            min_y = std::min(min_y, room.y);
            max_x = std::max(max_x, room.x + room.width);
            max_y = std::max(max_y, room.y + room.length);
            total_room_area += room.area();
        }
        
        double bbox_area = (max_x - min_x) * (max_y - min_y);
        double wasted_space = bbox_area - total_room_area;
        cost += wasted_space;
        
        chrome.fitness = cost + penalty;
        return chrome.fitness;
    }

    double calculateFitnessScore(double fitness) {
        double theoretical_min = 632.5;  // sum of min areas with cost multipliers
        return 100.0 * (theoretical_min / fitness);
    }
    
    Chromosome& tournamentSelect() {
        std::vector<int> indices(populationSize);
        for (int i = 0; i < populationSize; i++) indices[i] = i;
        std::shuffle(indices.begin(), indices.end(), gen);
        
        int bestIdx = indices[0];
        for (int i = 1; i < tournamentSize; i++) {
            if (population[indices[i]].fitness < population[bestIdx].fitness) {
                bestIdx = indices[i];
            }
        }
        return population[bestIdx];
    }
    
    std::pair<Chromosome, Chromosome> crossover(
        const Chromosome& parent1, const Chromosome& parent2)
    {
        Chromosome child1(roomSpecs.size()), child2(roomSpecs.size());
        
        std::uniform_real_distribution<> dist(0, 1);
        
        if (dist(gen) > crossoverRate) {
            child1 = parent1;
            child2 = parent2;
        } else {
            for (size_t i = 0; i < roomSpecs.size(); i++) {
                if (dist(gen) < 0.5) {
                    child1.rooms[i] = parent1.rooms[i];
                    child2.rooms[i] = parent2.rooms[i];
                } else {
                    child1.rooms[i] = parent2.rooms[i];
                    child2.rooms[i] = parent1.rooms[i];
                }
            }
        }
        
        return std::make_pair(child1, child2);
    }
    
    void mutate(Chromosome& chrome) {
        std::uniform_real_distribution<> probDist(0, 1);
        
        for (size_t i = 0; i < chrome.rooms.size(); i++) {
            Room& room = chrome.rooms[i];
            const RoomSpec& spec = roomSpecs[i];
            
            if (probDist(gen) < mutationRate) {
                std::normal_distribution<> lengthMut(0, 2.0);
                room.length += lengthMut(gen);
                room.length = std::max(spec.minLength, std::min(spec.maxLength, room.length));
            }
            
            if (probDist(gen) < mutationRate) {
                std::normal_distribution<> widthMut(0, 2.0);
                room.width += widthMut(gen);
                room.width = std::max(spec.minWidth, std::min(spec.maxWidth, room.width));
            }
            
            if (spec.aspectRatio > 0 && probDist(gen) < mutationRate / 2) {
                double avgDim = sqrt(room.area());
                room.width = avgDim / sqrt(spec.aspectRatio);
                room.length = avgDim * sqrt(spec.aspectRatio);
            }
            
            if (probDist(gen) < mutationRate) {
                std::normal_distribution<> posMut(0, 5.0);
                room.x += posMut(gen);
                room.x = std::max(0.0, room.x);
            }
            
            if (probDist(gen) < mutationRate) {
                std::normal_distribution<> posMut(0, 5.0);
                room.y += posMut(gen);
                room.y = std::max(0.0, room.y);
            }
        }
    }
    
    void initializePopulation() {
        population.clear();
        population.reserve(populationSize);
        
        for (int i = 0; i < populationSize; i++) {
            Chromosome chrome = createRandomChromosome();
            calculateFitness(chrome);
            population.push_back(chrome);
        }
        
        std::sort(population.begin(), population.end());
        bestSolution = population[0];
    }
    
    void runGeneration() {
        std::vector<Chromosome> newPopulation;
        newPopulation.reserve(populationSize);
        
        for (int i = 0; i < eliteCount; i++) {
            newPopulation.push_back(population[i]);
        }
        
        while (newPopulation.size() < static_cast<size_t>(populationSize)) {
            Chromosome& parent1 = tournamentSelect();
            Chromosome& parent2 = tournamentSelect();
            
            auto children = crossover(parent1, parent2);
            
            mutate(children.first);
            mutate(children.second);
            
            calculateFitness(children.first);
            calculateFitness(children.second);
            
            newPopulation.push_back(children.first);
            if (newPopulation.size() < static_cast<size_t>(populationSize)) {
                newPopulation.push_back(children.second);
            }
        }
        
        population = newPopulation;
        std::sort(population.begin(), population.end());
        
        if (population[0].fitness < bestSolution.fitness) {
            bestSolution = population[0];
        }
        
        bestFitnessHistory.push_back(population[0].fitness);
        double avgFitness = 0;
        for (const auto& ind : population) {
            avgFitness += ind.fitness;
        }
        avgFitnessHistory.push_back(avgFitness / populationSize);
    }
    
public:
    FloorplanGA(int popSize = 100, int gens = 200, double xRate = 0.8, 
                double mRate = 0.15, int elite = 5, int tournSize = 5)
        : populationSize(popSize), numGenerations(gens), crossoverRate(xRate),
          mutationRate(mRate), eliteCount(elite), tournamentSize(tournSize),
          bestSolution(7)
    {
        std::random_device rd;
        gen = std::mt19937(rd());
        initializeRoomSpecs();
    }
    
    void run() {
        std::cout << "Initializing Floorplan Genetic Algorithm...\n";
        std::cout << "Population Size: " << populationSize << "\n";
        std::cout << "Generations: " << numGenerations << "\n";
        std::cout << "Crossover Rate: " << crossoverRate << "\n";
        std::cout << "Mutation Rate: " << mutationRate << "\n\n";
        
        initializePopulation();
        
        std::cout << "Initial Best Fitness: " << std::fixed << std::setprecision(2) 
             << population[0].fitness << "\n\n";
        
        for (int gen = 0; gen < numGenerations; gen++) {
            runGeneration();
            
            if ((gen + 1) % 20 == 0 || gen == 0) {
                std::cout << "Generation " << std::setw(3) << (gen + 1) 
                     << " | Best: " << std::setw(8) << population[0].fitness
                     << " | Avg: " << std::setw(8) << avgFitnessHistory.back()
                     << " | Avg. Fitness " << calculateFitnessScore(bestSolution.fitness) << "\n";
                }
        }
        
        std::cout << "\n=== OPTIMIZATION COMPLETE ===\n\n";
    }
    
    void printBestSolution() {
        std::cout << "Best Solution Found:\n";
        std::cout << "Fitness: " << std::fixed << std::setprecision(2) << bestSolution.fitness << "\n\n";
        
        std::cout << std::setw(12) << "Room" << std::setw(10) << "Length" << std::setw(10) << "Width" 
             << std::setw(10) << "Area" << std::setw(10) << "X" << std::setw(10) << "Y" << "\n";
        std::cout << std::string(62, '-') << "\n";
        
        double totalArea = 0;
        double totalCost = 0;
        
        for (size_t i = 0; i < bestSolution.rooms.size(); i++) {
            const Room& room = bestSolution.rooms[i];
            const RoomSpec& spec = roomSpecs[i];
            double area = room.area();
            double cost = area * spec.costMultiplier;
            
            totalArea += area;
            totalCost += cost;
            
            std::cout << std::setw(12) << spec.name 
                 << std::setw(10) << room.length
                 << std::setw(10) << room.width
                 << std::setw(10) << area
                 << std::setw(10) << room.x
                 << std::setw(10) << room.y << "\n";
        }
        
        std::cout << std::string(62, '-') << "\n";
        std::cout << "Total Area: " << totalArea << "\n";
        std::cout << "Total Cost: " << totalCost << "\n\n";
        
        std::cout << "Constraint Validation:\n";
        bool allValid = true;
        
        for (size_t i = 0; i < bestSolution.rooms.size(); i++) {
            const Room& room = bestSolution.rooms[i];
            const RoomSpec& spec = roomSpecs[i];
            double area = room.area();
            
            bool valid = true;
            std::string issues;
            
            if (room.length < spec.minLength || room.length > spec.maxLength) {
                valid = false;
                issues += "Length out of range; ";
            }
            if (room.width < spec.minWidth || room.width > spec.maxWidth) {
                valid = false;
                issues += "Width out of range; ";
            }
            if (area < spec.minArea || area > spec.maxArea) {
                valid = false;
                issues += "Area out of range; ";
            }
            if (spec.aspectRatio > 0) {
                double ratio = room.length / room.width;
                if (std::abs(ratio - spec.aspectRatio) > 0.1) {
                    valid = false;
                    issues += "Aspect ratio violation; ";
                }
            }
            
            if (!valid) {
                std::cout << "  " << spec.name << ": FAILED - " << issues << "\n";
                allValid = false;
            }
        }
        
        for (size_t i = 0; i < bestSolution.rooms.size(); i++) {
            for (size_t j = i + 1; j < bestSolution.rooms.size(); j++) {
                double overlap = calculateOverlap(bestSolution.rooms[i], bestSolution.rooms[j]);
                if (overlap > 0.01) {
                    std::cout << "  Overlap between " << roomSpecs[i].name 
                         << " and " << roomSpecs[j].name << ": " << overlap << "\n";
                    allValid = false;
                }
            }
        }
        
        if (allValid) {
            std::cout << "  All constraints satisfied!\n";
        }
        std::cout << "\n";
    }
    
    void exportResults(const std::string& filename = "data/floorplan_results.csv") {
        FileExporter::exportFitnessHistory(bestFitnessHistory, avgFitnessHistory, filename);
        FileExporter::exportBestLayout(bestSolution, roomSpecs, "data/best_layout.csv");
    }
    
    const std::vector<double>& getBestFitnessHistory() const { return bestFitnessHistory; }
    const std::vector<double>& getAvgFitnessHistory() const { return avgFitnessHistory; }
};

#endif // FLOORPLANGA_H