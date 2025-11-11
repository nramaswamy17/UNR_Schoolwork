#ifndef GA_H
#define GA_H

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

#include "router.h"
using namespace std;

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

#endif GA_H