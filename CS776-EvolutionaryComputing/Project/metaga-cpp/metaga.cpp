#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <ctime>
#include <cmath>
#include <limits>
#include <sstream>
#include <filesystem>

#include "tour.h"
#include "graph.h"
#include "router.h"
#include "ga.h"

using namespace std;
namespace fs = std::filesystem;

// Return recursive list of .dat files in a directory and subdirectories (sorted)
vector<string> list_dat_files(const string& dir, bool test) {
    vector<string> files;
    if (test == false) {
        try {
            //for (const auto& entry : fs::directory_iterator(dir)) {
            for (const auto& entry : fs::recursive_directory_iterator(dir)) {
                if (entry.is_regular_file() && entry.path().extension() == ".dat") {
                    files.push_back(entry.path().string());
                }
            }
        } catch (const std::exception& e) {
            cerr << "Warning: could not read directory '" << dir << "': " << e.what() << "\n";
        }
        sort(files.begin(), files.end());
    } else {
        cout << "TEST MODE\n";
        files = {"benchmarks/kshs/kshs1.dat"};
    }
    return files;
}

int main() {
    // Benchmarks: populate from benchmarks/ directory (all .dat files)
    bool test = true;
    vector<string> instances = list_dat_files("benchmarks", test);

    vector<int> depots = {0, 0, 0, 0};  // 4 robots at depot 0 (vertex 1)
    vector<int> seeds = {8115,3520,8647,9420,3116,6377,6207,4187,3641,8591,3580,8524,2650,2811,9963,7537,3472,3714,8158,7284,6948,6119,5253,5134,7350,2652,9968,3914,6899,4715};
    //vector<int> seeds = {8115};

    for(string instance : instances) {
        Graph graph(instance);
        cout << "Loaded " << instance << ": " << graph.nVertices << " vertices, " << graph.nEdges << " edges" << endl;
        
        int geneLen = 2;
        int chromLen = graph.nEdges * geneLen;
        
        string basename = instance.substr(instance.find_last_of("/")+1);
        basename = basename.substr(0, basename.find_last_of("."));
        
        // Open single CSV file for this instance
        string logfile = "results/" + basename + ".csv";
        ofstream log(logfile);
        log << "seed,generation,objective" << endl;
        
        vector<double> bestObjectives;
        
        for(int seed : seeds) {
            cout << "Running " << basename << " seed " << seed << "... ";
            
            Router router(graph, depots, seed);
            MetaGA ga(geneLen, chromLen, seed);
            
            auto start = clock();
            ga.run(router, log, seed);
            double elapsed = (clock() - start) / (double)CLOCKS_PER_SEC;
            
            double bestObj = 1.0/ga.bestFitness;
            bestObjectives.push_back(bestObj);
            
            cout << "Best: " << bestObj << " in " << elapsed << "s" << endl;
        }
        
        log.close();
        
        // Write summary
        ofstream summary("results/" + basename + "_summary.txt");
        summary << "Instance: " << basename << endl;
        summary << "Seeds: ";
        for(int i = 0; i < seeds.size(); i++) {
            summary << seeds[i];
            if(i < seeds.size()-1) summary << ", ";
        }
        summary << endl;
        for(int i = 0; i < seeds.size(); i++) {
            summary << "  Seed " << seeds[i] << ": " << bestObjectives[i] << endl;
        }
        double mean = 0;
        for(double obj : bestObjectives) mean += obj;
        mean /= bestObjectives.size();
        summary << "Mean: " << mean << endl;
        summary.close();
        
        cout << endl;
    }
    
    return 0;
}