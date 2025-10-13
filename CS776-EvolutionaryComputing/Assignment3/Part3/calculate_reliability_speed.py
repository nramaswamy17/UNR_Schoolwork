#!/usr/bin/env python3
"""
Calculate Reliability and Speed metrics for TSP GA experiments
Generates data for LaTeX table
"""

import os
import glob
import numpy as np
from pathlib import Path

# Optimal tour lengths for benchmarks
OPTIMAL_LENGTHS = {
    'burma14': 3323,
    'berlin52': 7542,
    'eil51': 426,
    'eil76': 538,
    'lin105': 14379,
    'lin318': 42029
}

def parse_best_tour_file(filename):
    """Extract best tour length and evaluation number from tour file"""
    with open(filename, 'r') as f:
        lines = f.readlines()
        tour_length = int(lines[0].split(':')[1].strip())
        evaluation = int(lines[1].split(':')[1].strip())
        return tour_length, evaluation

def parse_2opt_results_file(filename):
    """Extract 2-opt improvement results"""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            ga_length = int(lines[0].split(':')[1].strip())
            twoopt_length = int(lines[1].split(':')[1].strip())
            return ga_length, twoopt_length
    except Exception as e:
        print(f"Warning: Could not parse 2-opt file {filename}: {e}")
        return None, None

def calculate_reliability_speed(experiment_dir, benchmark_name):
    """Calculate reliability and speed metrics for a benchmark"""
    
    optimal_length = OPTIMAL_LENGTHS[benchmark_name]
    
    # Find all tour files
    tour_pattern = os.path.join(experiment_dir, f'{benchmark_name}_best_tour_seed*.txt')
    tour_files = sorted(glob.glob(tour_pattern))
    
    if not tour_files:
        print(f"No tour files found for {benchmark_name}")
        return None
    
    # Collect GA results
    ga_lengths = []
    ga_evaluations = []
    
    for tour_file in tour_files:
        try:
            length, evaluation = parse_best_tour_file(tour_file)
            ga_lengths.append(length)
            ga_evaluations.append(evaluation)
        except Exception as e:
            print(f"Error reading {tour_file}: {e}")
            continue
    
    if not ga_lengths:
        return None
    
    # Calculate GA Quality and threshold
    avg_ga_length = np.mean(ga_lengths)
    ga_quality_percent = ((avg_ga_length - optimal_length) / optimal_length) * 100
    ga_quality_threshold = optimal_length * (1 + ga_quality_percent / 100)
    
    # Calculate GA Reliability
    ga_within_quality = [i for i, length in enumerate(ga_lengths) 
                         if length <= ga_quality_threshold]
    ga_reliability = (len(ga_within_quality) / len(ga_lengths)) * 100
    
    # Calculate GA Speed
    if ga_within_quality:
        ga_speed = np.mean([ga_evaluations[i] for i in ga_within_quality])
    else:
        ga_speed = 0
    
    # Find all 2-opt results files
    twoopt_pattern = os.path.join(experiment_dir, f'{benchmark_name}_2opt_results_seed*.txt')
    twoopt_files = sorted(glob.glob(twoopt_pattern))
    
    # Initialize 2-opt metrics
    twoopt_reliability = None
    twoopt_speed = None
    
    if twoopt_files:
        # Collect 2-opt results
        twoopt_lengths = []
        twoopt_ga_lengths = []
        
        for twoopt_file in twoopt_files:
            ga_len, twoopt_len = parse_2opt_results_file(twoopt_file)
            if ga_len is not None and twoopt_len is not None:
                twoopt_ga_lengths.append(ga_len)
                twoopt_lengths.append(twoopt_len)
        
        if twoopt_lengths:
            # Calculate 2-opt Quality and threshold
            avg_twoopt_length = np.mean(twoopt_lengths)
            twoopt_quality_percent = ((avg_twoopt_length - optimal_length) / optimal_length) * 100
            twoopt_quality_threshold = optimal_length * (1 + twoopt_quality_percent / 100)
            
            # Calculate 2-opt Reliability
            twoopt_within_quality = [i for i, length in enumerate(twoopt_lengths) 
                                     if length <= twoopt_quality_threshold]
            twoopt_reliability = (len(twoopt_within_quality) / len(twoopt_lengths)) * 100
            
            # Calculate 2-opt Speed
            # Speed is based on when GA found the solution (before 2-opt was applied)
            if twoopt_within_quality:
                twoopt_speed = np.mean([ga_evaluations[i] for i in twoopt_within_quality 
                                       if i < len(ga_evaluations)])
            else:
                twoopt_speed = 0
    
    return {
        'benchmark': benchmark_name,
        'optimal_length': optimal_length,
        'num_runs': len(ga_lengths),
        'ga_avg_length': avg_ga_length,
        'ga_quality_percent': ga_quality_percent,
        'ga_quality_threshold': ga_quality_threshold,
        'ga_reliability': ga_reliability,
        'ga_speed': ga_speed,
        'twoopt_reliability': twoopt_reliability,
        'twoopt_speed': twoopt_speed
    }

def print_latex_table(results):
    """Print results in LaTeX table format"""
    
    print("\n" + "="*80)
    print("LATEX TABLE OUTPUT")
    print("="*80 + "\n")
    
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\caption{Reliability and Speed (R$\\ge$30). Reliability is \\% of runs meeting the Quality threshold; Speed is mean evaluations to first meet it.}")
    print("\\label{tab:tsp_rel_speed}")
    print("\\begin{tabular}{lrrrr}")
    print("\\toprule")
    print("Instance & Reliability$_\\mathrm{GA}$ (\\%) & Reliability$_{\\mathrm{GA}\\to\\mathrm{2\\text{-}Opt}}$ (\\%) & Speed$_\\mathrm{GA}$ (evals) & Speed$_{\\mathrm{GA}\\to\\mathrm{2\\text{-}Opt}}$ (evals) \\\\")
    print("\\midrule")
    
    for result in results:
        benchmark = result['benchmark']
        ga_rel = result['ga_reliability']
        ga_speed = result['ga_speed']
        twoopt_rel = result['twoopt_reliability']
        twoopt_speed = result['twoopt_speed']
        
        # Format values
        ga_rel_str = f"{ga_rel:.1f}" if ga_rel is not None else "N/A"
        ga_speed_str = f"{ga_speed:.0f}" if ga_speed is not None and ga_speed > 0 else "N/A"
        
        if twoopt_rel is not None:
            twoopt_rel_str = f"{twoopt_rel:.1f}"
        else:
            twoopt_rel_str = "N/A"
        
        if twoopt_speed is not None and twoopt_speed > 0:
            twoopt_speed_str = f"{twoopt_speed:.0f}"
        else:
            twoopt_speed_str = "N/A"
        
        print(f"\\texttt{{{benchmark:8s}}} & {ga_rel_str:>6s} & {twoopt_rel_str:>6s} & {ga_speed_str:>6s} & {twoopt_speed_str:>6s} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()

def print_detailed_results(results):
    """Print detailed results for verification"""
    
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80 + "\n")
    
    for result in results:
        print(f"Benchmark: {result['benchmark']}")
        print(f"  Optimal Length: {result['optimal_length']}")
        print(f"  Number of Runs: {result['num_runs']}")
        print(f"\n  GA Results:")
        print(f"    Average Best Length: {result['ga_avg_length']:.2f}")
        print(f"    Quality (% from optimal): {result['ga_quality_percent']:.2f}%")
        print(f"    Quality Threshold: {result['ga_quality_threshold']:.2f}")
        print(f"    Reliability: {result['ga_reliability']:.2f}%")
        print(f"    Speed: {result['ga_speed']:.0f} evaluations")
        
        if result['twoopt_reliability'] is not None:
            print(f"\n  GA→2-Opt Results:")
            print(f"    Reliability: {result['twoopt_reliability']:.2f}%")
            print(f"    Speed: {result['twoopt_speed']:.0f} evaluations")
        else:
            print(f"\n  GA→2-Opt Results: Not available")
        
        print(f"\n{'-'*80}\n")

def main():
    """Main function to calculate reliability and speed for all benchmarks"""
    
    base_data_dir = 'data'
    
    # Find all experiment directories
    experiment_dirs = []
    for benchmark in OPTIMAL_LENGTHS.keys():
        exp_dir = os.path.join(base_data_dir, f'{benchmark}_experiments')
        if os.path.exists(exp_dir):
            experiment_dirs.append((benchmark, exp_dir))
    
    if not experiment_dirs:
        print(f"No experiment directories found in {base_data_dir}")
        print("Looking for directories like: berlin52_experiments, burma14_experiments, etc.")
        return
    
    print(f"Found {len(experiment_dirs)} experiment directories")
    print("="*80)
    
    # Process each benchmark
    results = []
    for benchmark_name, exp_dir in experiment_dirs:
        print(f"\nProcessing {benchmark_name}...")
        
        result = calculate_reliability_speed(exp_dir, benchmark_name)
        
        if result is None:
            print(f"Skipping {benchmark_name} - no valid data found")
            continue
        
        results.append(result)
        print(f"Completed {benchmark_name}")
    
    # Sort results by benchmark name
    results.sort(key=lambda x: x['benchmark'])
    
    # Print detailed results
    print_detailed_results(results)
    
    # Print LaTeX table
    print_latex_table(results)
    
    # Save to file
    output_file = 'reliability_speed_table.txt'
    with open(output_file, 'w') as f:
        # Redirect print to file
        import sys
        old_stdout = sys.stdout
        sys.stdout = f
        print_latex_table(results)
        sys.stdout = old_stdout
    
    print(f"LaTeX table also saved to: {output_file}")
    print("="*80)

if __name__ == '__main__':
    main()