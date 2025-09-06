#include "dejong_functions.h"
#include <iostream>
#include <stdexcept>

// Function implementation
double sphere(std::vector<double>& vec){
    double sum = 0.0;
    //std::cout << "Size of vector: " << vec.size() << std::endl;
    for (size_t i = 0; i < vec.size(); i++){
        if (vec[i] < -5.12 || vec[i] > 5.12){
            throw std::out_of_range("Vector element out of bounds [-5.12, +5.12]. Ensure all vector elements are within range");
        }
        sum += vec[i] * vec[i];
    }
    return sum;
}

double rosenbrock(std::vector<double>& vec){
    double sum = 0.0;
    //std::cout << "Size of vector: " << vec.size() << std::endl;
    for (size_t i = 0; i < vec.size()-1; i++){
        if (vec[i] < -2.048 || vec[i] > 2.048){
            throw std::out_of_range("Vector element out of bounds [-2.048, +2.048]. Ensure all vector elements are within range");
        }
        sum += 100 * std::pow(vec[i+1] - vec[i]*vec[i], 2) + std::pow(vec[i] - 1, 2);
    }
    return sum;
}

double step(std::vector<double>& vec) {
    double sum = 0.0;
    //std::cout << "Size of vector: " << vec.size() << std::endl;
    for (size_t i = 0; i < vec.size(); i++){
        if (vec[i] < -5.12 || vec[i] > 5.12){
            throw std::out_of_range("Vector element out of bounds [-5.12, +5.12]. Ensure all vector elements are within range");
        }
        sum += std::floor(vec[i]);
    }
    return sum;
}