#!/bin/bash

# Create results directory
mkdir -p results

# clear results directory
#rm -rf results/*

# Create build directory
mkdir -p build

# Clear build directory
rm -rf build/*

# Compile C++ code
echo "Compiling..."
g++ -O3 -std=c++17 metaga.cpp -o build/metaga

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful!"
echo ""

echo "Running MetaGA..."
./build/metaga

if [ $? -ne 0 ]; then
    echo "Execution failed!"
    exit 1
fi

echo ""
echo "Generating visualizations..."
python3 visualize.py

echo ""
echo "Done! Check results/ directory for output."