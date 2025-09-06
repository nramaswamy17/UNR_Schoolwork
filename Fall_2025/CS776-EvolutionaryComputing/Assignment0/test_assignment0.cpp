#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "dejong_functions.h"

void testSphereFunction() {
    std::cout <<"==================================" <<std::endl;
    std::cout << "Running sphere function tests..." << std::endl;
    
    // Test 1: Global Minimmum vector (should return 0)
    std::vector<double> zero_vec = {0, 0, 0};
    double result = sphere(zero_vec);
    assert(result == 0.0);
    std::cout << "✓ Global Minimmum vector test passed: " << result << std::endl;
    
    // Test 2: Unit vector (should return 1)
    std::vector<double> unit_vec = {1, 0, 0};
    result = sphere(unit_vec);
    assert(result == 1.0);
    std::cout << "✓ Unit vector test passed: " << result << std::endl;
    
    // Test 3: Multiple values
    std::vector<double> test_vec = {1, 2, 3};
    result = sphere(test_vec);
    assert(result == 14.0); // 1² + 2² + 3² = 1 + 4 + 9 = 14
    std::cout << "✓ Multiple values test passed: " << result << std::endl;
    
    // Test 4: Boundary values
    std::vector<double> boundary_vec = {5.12, -5.12, 0};
    result = sphere(boundary_vec);
    assert(result == 52.4288); // 5.12² + (-5.12)² + 0² = 26.2144 + 26.2144 + 0
    std::cout << "✓ Boundary values test passed: " << result << std::endl;
    
    // Test 5: Out of bounds (should throw exception)
    std::vector<double> out_of_bounds = {6.0, 0, 0};
    try {
        result = sphere(out_of_bounds);
        std::cout << "✗ Out of bounds test failed - should have thrown exception" << std::endl;
    } catch (const std::out_of_range& e) {
        std::cout << "✓ Out of bounds test passed - exception caught: " << e.what() << std::endl;
    }
    
    std::cout << "All tests completed!" << std::endl;
    std::cout <<"==================================" <<std::endl;
}

void testRosenbrockFunction() {
    std::cout <<"==================================" <<std::endl;
    std::cout << "Running Rosenbrock function tests..." << std::endl;
    
    // Test 1: Global Minimmum vector (should return 0)
    std::vector<double> zero_vec = {1,1};
    double result = rosenbrock(zero_vec);
    assert(result == 0.0);
    std::cout << "✓ Global Minimmum vector test passed: " << result << std::endl;

    // Test 2: Boundary values
    std::vector<double> boundary_vec = {2.048, -2.048};
    result = rosenbrock(boundary_vec);
    assert(std::abs(result - 3897.73) < .01); 
    std::cout << "✓ Boundary values test passed: " << result << std::endl;

    std::cout << "All tests completed!" << std::endl;
    std::cout <<"==================================" <<std::endl;
}

void testStepFunction(){
    std::cout <<"==================================" <<std::endl;
    std::cout << "Running Step function tests..." << std::endl;
    
    // Test 1: Global Minimmum vector (should return 0)
    std::vector<double> zero_vec = {-5.12, -5.12, -5.12, -5.12, -5.12};
    double result = step(zero_vec);
    assert(std::abs(result + 30) < .01); // floor(-5.12) = 6 => 6 * -5 = -30
    std::cout << "✓ Global Minimmum vector test passed: " << result << std::endl;

    // Test 2: Boundary values (mix of -5.12 and 5.12)
    std::vector<double> boundary_vec = {-5.12, -5.12, -5.12, -5.12, -5.12};
    result = step(boundary_vec);
    assert(std::abs(result + 30) < .01); 
    std::cout << "✓ Boundary values test passed: " << result << std::endl;
    
    // Test 3: Out of bounds (should throw exception)
    std::vector<double> out_of_bounds = {6.0, 0, 0, 0, 0};
    try {
        result = step(out_of_bounds);
        std::cout << "✗ Out of bounds test failed - should have thrown exception" << std::endl;
    } catch (const std::out_of_range& e) {
        std::cout << "✓ Out of bounds test passed - exception caught: " << e.what() << std::endl;
    }

    std::cout << "All tests completed!" << std::endl;
    std::cout <<"==================================" <<std::endl;
}

int main() {
    testSphereFunction();
    testRosenbrockFunction();
    testStepFunction();
    return 0;
}
