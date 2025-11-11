#ifndef TOUR_H
#define TOUR_H

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

#include "graph.h"
using namespace std;

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

#endif TOUR_H