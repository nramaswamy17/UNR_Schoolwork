#include "FloorplanGA.h"
#include <iostream>

int main() {
    // Create and run GA with final parameters from report
    FloorplanGA ga(
        200,   // population size
        2000,   // generations
        0.85,  // crossover rate
        0.02,  // mutation rate
        10,     // elite count
        7,      // tournament size
        1       // seed
    );
    
    ga.run();
    ga.printBestSolution();
    //ga.exportResults();
    return 0;
}