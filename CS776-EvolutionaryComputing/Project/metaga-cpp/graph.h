#ifndef GRAPH_H
#define GRAPH_H

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

#endif GRAPH_H