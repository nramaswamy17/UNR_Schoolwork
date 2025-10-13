#ifndef TSP_INSTANCE_H
#define TSP_INSTANCE_H

#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

struct City {
    int id;
    double x;
    double y;
};

enum EdgeWeightType {
    EUC_2D,
    GEO,
    UNKNOWN
};

class TSPInstance {
private:
    std::string name;
    std::vector<City> cities;
    std::vector<std::vector<int>> distanceMatrix;
    int optimalTourLength;
    EdgeWeightType weightType;
    
    int euclideanDistance(const City& a, const City& b) const {
        double dx = a.x - b.x;
        double dy = a.y - b.y;
        double dist = sqrt(dx * dx + dy * dy);
        return static_cast<int>(dist + 0.5); // Round to nearest integer (TSPLIB standard)
    }
    
    int geoDistance(const City& a, const City& b) const {
        // TSPLIB GEO distance formula for geographical coordinates
        const double PI = 3.141592653589793;
        const double RRR = 6378.388; // Earth radius in km
        
        // Convert to radians
        double deg_a_x = static_cast<int>(a.x);
        double min_a_x = a.x - deg_a_x;
        double lat_a = PI * (deg_a_x + 5.0 * min_a_x / 3.0) / 180.0;
        
        double deg_a_y = static_cast<int>(a.y);
        double min_a_y = a.y - deg_a_y;
        double lon_a = PI * (deg_a_y + 5.0 * min_a_y / 3.0) / 180.0;
        
        double deg_b_x = static_cast<int>(b.x);
        double min_b_x = b.x - deg_b_x;
        double lat_b = PI * (deg_b_x + 5.0 * min_b_x / 3.0) / 180.0;
        
        double deg_b_y = static_cast<int>(b.y);
        double min_b_y = b.y - deg_b_y;
        double lon_b = PI * (deg_b_y + 5.0 * min_b_y / 3.0) / 180.0;
        
        double q1 = cos(lon_a - lon_b);
        double q2 = cos(lat_a - lat_b);
        double q3 = cos(lat_a + lat_b);
        
        double distance = RRR * acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0;
        
        return static_cast<int>(distance);
    }
    
    void computeDistanceMatrix() {
        int n = cities.size();
        distanceMatrix.resize(n, std::vector<int>(n, 0));
        
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int dist;
                if (weightType == GEO) {
                    dist = geoDistance(cities[i], cities[j]);
                } else {
                    dist = euclideanDistance(cities[i], cities[j]);
                }
                distanceMatrix[i][j] = dist;
                distanceMatrix[j][i] = dist;
            }
        }
    }

public:
    TSPInstance() : optimalTourLength(-1), weightType(EUC_2D) {}
    
    bool loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return false;
        }
        
        std::string line;
        bool nodeSection = false;
        
        // Parse header
        while (std::getline(file, line)) {
            if (line.find("NAME") != std::string::npos) {
                size_t colonPos = line.find(':');
                if (colonPos != std::string::npos) {
                    name = line.substr(colonPos + 1);
                    // Trim whitespace
                    name.erase(0, name.find_first_not_of(" \t\r\n"));
                    name.erase(name.find_last_not_of(" \t\r\n") + 1);
                }
            }
            else if (line.find("EDGE_WEIGHT_TYPE") != std::string::npos) {
                if (line.find("GEO") != std::string::npos) {
                    weightType = GEO;
                    std::cout << "Detected edge weight type: GEO (geographical)" << std::endl;
                } else if (line.find("EUC_2D") != std::string::npos) {
                    weightType = EUC_2D;
                    std::cout << "Detected edge weight type: EUC_2D (euclidean)" << std::endl;
                } else {
                    weightType = EUC_2D; // Default
                    std::cout << "Using default edge weight type: EUC_2D" << std::endl;
                }
            }
            else if (line.find("NODE_COORD_SECTION") != std::string::npos) {
                nodeSection = true;
                break;
            }
        }
        
        if (!nodeSection) {
            std::cerr << "Error: NODE_COORD_SECTION not found" << std::endl;
            return false;
        }
        
        // Parse cities
        while (std::getline(file, line)) {
            if (line.find("EOF") != std::string::npos) {
                break;
            }
            
            std::istringstream iss(line);
            City city;
            if (iss >> city.id >> city.x >> city.y) {
                cities.push_back(city);
            }
        }
        
        file.close();
        
        if (cities.empty()) {
            std::cerr << "Error: No cities loaded" << std::endl;
            return false;
        }
        
        computeDistanceMatrix();
        std::cout << "Loaded TSP instance: " << name << " with " << cities.size() << " cities" << std::endl;
        
        return true;
    }
    
    void setOptimalTourLength(int length) { optimalTourLength = length; }
    
    int getNumCities() const { return cities.size(); }
    const std::vector<City>& getCities() const { return cities; }
    int getDistance(int cityA, int cityB) const {
        return distanceMatrix[cityA][cityB];
    }
    int getOptimalTourLength() const { return optimalTourLength; }
    std::string getName() const { return name; }
};

#endif