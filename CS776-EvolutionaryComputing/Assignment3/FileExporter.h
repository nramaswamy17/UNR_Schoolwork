#ifndef FILEEXPORTER_H
#define FILEEXPORTER_H

#include "Chromosome.h"
#include "RoomSpec.h"
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

class FileExporter {
public:
    static void exportFitnessHistory(
        const std::vector<double>& bestFitness,
        const std::vector<double>& avgFitness,
        const std::string& filename = "data/floorplan_results.csv")
    {
        std::ofstream file(filename);
        
        if (!file.is_open()) {
            std::cerr << "Error opening file for export: " << filename << "\n";
            return;
        }
        
        file << "Generation,Best_Fitness,Avg_Fitness\n";
        for (size_t i = 0; i < bestFitness.size(); i++) {
            file << i << "," << bestFitness[i] << "," << avgFitness[i] << "\n";
        }
        
        file.close();
        std::cout << "Results exported to " << filename << "\n";
    }
    
    static void exportBestLayout(
        const Chromosome& solution,
        const std::vector<RoomSpec>& specs,
        const std::string& filename = "data/best_layout.csv")
    {
        std::ofstream file(filename);
        
        if (!file.is_open()) {
            std::cerr << "Error opening file for export: " << filename << "\n";
            return;
        }
        
        file << "Room,Length,Width,Area,X,Y\n";
        for (size_t i = 0; i < solution.rooms.size(); i++) {
            const Room& room = solution.rooms[i];
            file << specs[i].name << ","
                 << room.length << ","
                 << room.width << ","
                 << room.area() << ","
                 << room.x << ","
                 << room.y << "\n";
        }
        
        file.close();
        std::cout << "Best layout exported to " << filename << "\n";
    }
};

#endif // FILEEXPORTER_H