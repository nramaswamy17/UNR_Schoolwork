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

// Graph class
class Graph {
public:
    vector<vector<double>> adjMatrix;
    vector<pair<int,int>> edgeIds;
    int nVertices, nEdges;
    
    Graph(string filepath) {
        ifstream file(filepath);
        string line;
        
        while(getline(file, line)) {
            if(line.find("VERTICES") != string::npos) {
                sscanf(line.c_str(), " VERTICES : %d", &nVertices);
                adjMatrix.resize(nVertices, vector<double>(nVertices, 0));
            }
            if(line.find("LISTA_ARISTAS_REQ") != string::npos) break;
        }
        
        nEdges = 0;
        while(getline(file, line)) {
            if(line.find("DEPOSITO") != string::npos) break;
            int v1, v2;
            double cost;
            if(sscanf(line.c_str(), " ( %d, %d) coste %lf", &v1, &v2, &cost) == 3) {
                v1--; v2--;
                adjMatrix[v1][v2] = adjMatrix[v2][v1] = cost;
                edgeIds.push_back({v1, v2});
                nEdges++;
            }
        }
        file.close();
        computeShortestPaths();
    }
    
    vector<vector<double>> dist;
    vector<vector<vector<int>>> paths;
    
    void computeShortestPaths() {
        dist = adjMatrix;
        paths.resize(nVertices, vector<vector<int>>(nVertices));
        
        for(int i = 0; i < nVertices; i++) {
            for(int j = 0; j < nVertices; j++) {
                if(i == j) dist[i][j] = 0;
                else if(dist[i][j] == 0) dist[i][j] = 1e9;
                paths[i][j] = {i, j};
            }
        }
        
        for(int k = 0; k < nVertices; k++) {
            for(int i = 0; i < nVertices; i++) {
                for(int j = 0; j < nVertices; j++) {
                    if(dist[i][k] + dist[k][j] < dist[i][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j];
                        paths[i][j] = paths[i][k];
                        paths[i][j].insert(paths[i][j].end(), paths[k][j].begin()+1, paths[k][j].end());
                    }
                }
            }
        }
    }
    
    double getEdgeCost(int e) { return adjMatrix[edgeIds[e].first][edgeIds[e].second]; }
};

// Tour class
class Tour {
public:
    vector<int> vertices, edges;
    double cost;
    int depot;
    
    Tour() : cost(0), depot(0) {}
    
    void clear() { vertices.clear(); edges.clear(); cost = 0; }
    
    void addVertex(int v, Graph& g) {
        if(vertices.empty()) {
            vertices.push_back(v);
        } else if(vertices.back() != v) {
            cost += g.dist[vertices.back()][v];
            vertices.push_back(v);
        }
    }
    
    void addEdge(int e, Graph& g) {
        int v1 = g.edgeIds[e].first, v2 = g.edgeIds[e].second;
        if(vertices.empty()) {
            vertices.push_back(v1);
            vertices.push_back(v2);
            edges.push_back(e);
            cost += g.getEdgeCost(e);
        } else {
            int last = vertices.back();
            int target = (last == v1) ? v2 : v1;
            if(last != v1 && last != v2) {
                double d1 = g.dist[last][v1], d2 = g.dist[last][v2];
                target = (d1 < d2) ? v1 : v2;
                cost += min(d1, d2);
            }
            vertices.push_back(target);
            edges.push_back(e);
            cost += g.getEdgeCost(e);
        }
    }
};

// Router class
class Router {
public:
    Graph& graph;
    vector<Tour> tours;
    vector<int> depots;
    vector<bool> visited;
    mt19937 rng;
    
    Router(Graph& g, vector<int> deps, int seed) : graph(g), depots(deps), rng(seed) {
        tours.resize(depots.size());
        for(int i = 0; i < tours.size(); i++) tours[i].depot = depots[i];
        visited.resize(g.nEdges, false);
    }
    
    void clear() {
        for(auto& t : tours) t.clear();
        fill(visited.begin(), visited.end(), false);
    }
    
    Tour& getShortestTour() {
        return *min_element(tours.begin(), tours.end(), [](Tour& a, Tour& b) { return a.cost < b.cost; });
    }
    
    double getMaxTourCost() {
        double mx = 0;
        for(auto& t : tours) mx = max(mx, t.cost);
        return mx;
    }
    
    vector<int> getNearestUnvisitedEdges(int v) {
        vector<pair<double,int>> dists;
        for(int e = 0; e < graph.nEdges; e++) {
            if(!visited[e]) {
                int v1 = graph.edgeIds[e].first, v2 = graph.edgeIds[e].second;
                double d = min(graph.dist[v][v1], graph.dist[v][v2]);
                dists.push_back({d, e});
            }
        }
        sort(dists.begin(), dists.end());
        
        vector<int> result;
        if(dists.empty()) return result;
        
        double minDist = dists[0].first;
        for(auto& p : dists) {
            if(p.first == minDist) result.push_back(p.second);
            else break;
        }
        
        sort(result.begin(), result.end(), [&](int a, int b) { 
            return graph.getEdgeCost(a) < graph.getEdgeCost(b); 
        });
        return result;
    }
    
    void extendTourToEdge(int e, Tour& t) {
        int prevSize = t.edges.size();
        t.addEdge(e, graph);
        for(int i = prevSize; i < t.edges.size(); i++) visited[t.edges[i]] = true;
    }
    
    void heuristic0(int h) { // min
        auto& t = getShortestTour();
        auto edges = getNearestUnvisitedEdges(t.vertices.back());
        if(!edges.empty()) extendTourToEdge(edges[0], t);
    }
    
    void heuristic1(int h) { // median
        auto& t = getShortestTour();
        auto edges = getNearestUnvisitedEdges(t.vertices.back());
        if(!edges.empty()) extendTourToEdge(edges[edges.size()/2], t);
    }
    
    void heuristic2(int h) { // max
        auto& t = getShortestTour();
        auto edges = getNearestUnvisitedEdges(t.vertices.back());
        if(!edges.empty()) extendTourToEdge(edges.back(), t);
    }
    
    void heuristic3(int h) { // random
        auto& t = getShortestTour();
        auto edges = getNearestUnvisitedEdges(t.vertices.back());
        if(!edges.empty()) {
            uniform_int_distribution<> dis(0, edges.size()-1);
            extendTourToEdge(edges[dis(rng)], t);
        }
    }
    
    void applyHeuristic(int h) {
        switch(h) {
            case 0: heuristic0(h); break;
            case 1: heuristic1(h); break;
            case 2: heuristic2(h); break;
            case 3: heuristic3(h); break;
        }
    }
};

// GA class
class MetaGA {
public:
    int geneLen, chromLen, popSize, numGens;
    vector<vector<int>> population;
    vector<double> fitness;
    mt19937 rng;
    double bestFitness;
    vector<int> bestChrom;
    int bestGen;
    
    MetaGA(int gl, int cl, int seed) : geneLen(gl), chromLen(cl), rng(seed), 
        popSize(100), numGens(150), bestFitness(0), bestGen(0) {}
    
    void initialize() {
        population.resize(popSize, vector<int>(chromLen));
        fitness.resize(popSize);
        uniform_int_distribution<> dis(0, 1);
        for(auto& ind : population)
            for(auto& gene : ind) gene = dis(rng);
    }
    
    double evaluate(vector<int>& chrom, Router& router) {
        vector<int> decoding;
        for(int i = 0; i < chromLen; i += geneLen) {
            int val = 0;
            for(int j = 0; j < geneLen; j++) val = val * 2 + chrom[i+j];
            decoding.push_back(val % 4);
        }
        
        router.clear();
        for(auto& t : router.tours) t.addVertex(t.depot, router.graph);
        
        for(int h : decoding) router.applyHeuristic(h);
        
        for(auto& t : router.tours) t.addVertex(t.depot, router.graph);
        
        double obj = router.getMaxTourCost();
        return 1.0 / obj;
    }
    
    void evaluatePopulation(Router& router) {
        for(int i = 0; i < popSize; i++) {
            fitness[i] = evaluate(population[i], router);
            if(fitness[i] > bestFitness) {
                bestFitness = fitness[i];
                bestChrom = population[i];
            }
        }
    }
    
    vector<int> tournamentSelect() {
        uniform_int_distribution<> dis(0, popSize-1);
        int i1 = dis(rng), i2 = dis(rng);
        return fitness[i1] > fitness[i2] ? population[i1] : population[i2];
    }
    
    void crossover(vector<int>& p1, vector<int>& p2, vector<int>& c) {
        uniform_int_distribution<> dis(0, chromLen-1);
        int pt1 = dis(rng), pt2 = dis(rng);
        if(pt1 > pt2) swap(pt1, pt2);
        
        c = p1;
        for(int i = pt1; i <= pt2; i++) c[i] = p2[i];
    }
    
    void mutate(vector<int>& ind) {
        uniform_real_distribution<> dis(0, 1);
        if(dis(rng) < 0.1) {
            uniform_int_distribution<> pos(0, chromLen-1);
            int i = pos(rng), j = pos(rng);
            if(i > j) swap(i, j);
            reverse(ind.begin()+i, ind.begin()+j+1);
        }
    }
    
    void evolve(Router& router) {
        vector<pair<double,int>> ranked;
        for(int i = 0; i < popSize; i++) ranked.push_back({fitness[i], i});
        sort(ranked.rbegin(), ranked.rend());
        
        vector<vector<int>> newPop;
        for(int i = 0; i < 50; i++) newPop.push_back(population[ranked[i].second]);
        
        while(newPop.size() < popSize) {
            auto p1 = tournamentSelect();
            auto p2 = tournamentSelect();
            vector<int> child;
            crossover(p1, p2, child);
            mutate(child);
            newPop.push_back(child);
        }
        
        population = newPop;
    }
    
    void run(Router& router, ofstream& log, int seed) {
        initialize();
        
        for(int gen = 0; gen < numGens; gen++) {
            evaluatePopulation(router);
            log << seed << "," << gen << "," << (1.0/bestFitness) << endl;
            if(gen % 10 == 0) cout << ".";
            if(gen < numGens-1) evolve(router);
        }
        cout << endl;
    }
};

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