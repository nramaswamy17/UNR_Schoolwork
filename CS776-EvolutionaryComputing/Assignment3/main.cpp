#include "FloorplanGA.h"
#include <iostream>

int main() {
    // Create and run GA with final parameters from report
    FloorplanGA ga(
        100,   // population size
        2000,   // generations
        0.30,  // crossover rate
        0.01,  // mutation rate
        5,     // elite count
        5      // tournament size
    );
    
    ga.run();
    //ga.printBestSolution();
    //ga.exportResults();
    return 0;
}