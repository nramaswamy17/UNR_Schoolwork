#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <random>
#include <string>
#include <fstream>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>

class Utils {
private:
    static std::mt19937& getRNG() {
        static std::mt19937 rng;
        return rng;
    }
    
public:
    static void setSeed(unsigned int seed) {
        getRNG().seed(seed);
    }
    
    static int randomInt(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(getRNG());
    }
    
    static double randomDouble(double min = 0.0, double max = 1.0) {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(getRNG());
    }
    
    static bool randomBool(double probability = 0.5) {
        return randomDouble() < probability;
    }
    
    template<typename T>
    static void shuffleVector(std::vector<T>& vec) {
        std::shuffle(vec.begin(), vec.end(), getRNG());
    }
    
    static bool fileExists(const std::string& filename) {
        std::ifstream file(filename);
        return file.good();
    }
    
    static void createDirectory(const std::string& path) {
#ifdef _WIN32
        _mkdir(path.c_str());
#else
        mkdir(path.c_str(), 0777);
#endif
    }
};

#endif