#!/bin/bash

# Script to run multiple experiments for TSP GA
# Usage: ./run_experiments.sh <benchmark> <num_runs>

if [ $# -lt 2 ]; then
    echo "Usage: $0 <benchmark> <num_runs>"
    echo "Example: $0 berlin52 30"
    echo ""
    echo "Available benchmarks:"
    echo "  burma14, berlin52, eil51, eil76, lin105, lin318"
    exit 1
fi

BENCHMARK=$1
NUM_RUNS=$2

echo "Running $NUM_RUNS experiments for $BENCHMARK"
echo "========================================"

# Create subdirectory for this benchmark
OUTPUT_DIR="data/${BENCHMARK}_experiments"
mkdir -p "$OUTPUT_DIR"

# Run experiments
for ((seed=1; seed<=NUM_RUNS; seed++)); do
    echo ""
    echo "Run $seed/$NUM_RUNS (seed=$seed)"
    echo "----------------------------------------"
    
    ./tsp_ga "$BENCHMARK" "$seed"
    
    # Move output files to experiment directory
    if [ -f "data/${BENCHMARK}_stats.csv" ]; then
        mv "data/${BENCHMARK}_stats.csv" "$OUTPUT_DIR/${BENCHMARK}_stats_seed${seed}.csv"
    fi
    
    if [ -f "data/${BENCHMARK}_best_tour.txt" ]; then
        mv "data/${BENCHMARK}_best_tour.txt" "$OUTPUT_DIR/${BENCHMARK}_best_tour_seed${seed}.txt"
    fi
    
    if [ -f "data/${BENCHMARK}_2opt_results.txt" ]; then
        mv "data/${BENCHMARK}_2opt_results.txt" "$OUTPUT_DIR/${BENCHMARK}_2opt_results_seed${seed}.txt"
    fi
done

echo ""
echo "========================================"
echo "All experiments completed!"
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "To analyze results, you can use:"
echo "  python3 analyze_results.py $OUTPUT_DIR"