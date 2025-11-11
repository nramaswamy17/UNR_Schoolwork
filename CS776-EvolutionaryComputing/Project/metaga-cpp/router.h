#ifndef ROUTER_H
#define ROUTER_H

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

#include "graph.h";
#include "tour.h";
using namespace std;

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

#endif ROUTER_H