#!/usr/bin/env python3
"""
Analyze TSP GA experiment results
Computes Quality, Reliability, and Speed metrics
"""

import os
import sys
import glob
import csv
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
            improvement = int(lines[2].split(':')[1].strip())
            percent_improvement = float(lines[3].split(':')[1].strip().replace('%', '').strip())
            num_swaps = int(lines[4].split(':')[1].strip())
            return ga_length, twoopt_length, improvement, percent_improvement, num_swaps
    except Exception as e:
        print(f"Warning: Could not parse 2-opt file {filename}: {e}")
        return None

def parse_stats_file(filename):
    """Extract generation statistics from CSV file"""
    generations = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            generations.append({
                'generation': int(row['Generation']),
                'best_length': int(row['BestLength']),
                'avg_length': float(row['AvgLength']),
                'evaluations': int(row['Evaluations'])
            })
    return generations

def analyze_experiment_directory(directory):
    """Analyze all experiment results in a directory"""
    
    # Extract benchmark name from directory
    benchmark_name = None
    for name in OPTIMAL_LENGTHS.keys():
        if name in directory:
            benchmark_name = name
            break
    
    if not benchmark_name:
        print(f"Warning: Could not determine benchmark name from {directory}")
        return None
    
    optimal_length = OPTIMAL_LENGTHS[benchmark_name]
    
    # Find all tour files
    tour_files = glob.glob(os.path.join(directory, '*_best_tour_seed*.txt'))
    stats_files = glob.glob(os.path.join(directory, '*_stats_seed*.csv'))
    twoopt_files = glob.glob(os.path.join(directory, '*_2opt_results_seed*.txt'))
    
    if not tour_files:
        print(f"No tour files found in {directory}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {benchmark_name}")
    print(f"Optimal tour length: {optimal_length}")
    print(f"Number of runs: {len(tour_files)}")
    print(f"{'='*60}")
    
    # Collect best tour lengths from all runs
    best_lengths = []
    evaluations_to_best = []
    
    for tour_file in sorted(tour_files):
        length, evaluation = parse_best_tour_file(tour_file)
        best_lengths.append(length)
        evaluations_to_best.append(evaluation)
    
    # Calculate metrics
    avg_best_length = np.mean(best_lengths)
    std_best_length = np.std(best_lengths)
    min_best_length = np.min(best_lengths)
    max_best_length = np.max(best_lengths)
    
    # Quality: percentage distance from optimum (average of best over all runs)
    quality_percent = ((avg_best_length - optimal_length) / optimal_length) * 100
    
    # For Reliability and Speed, we define "within Quality" as being within
    # quality_percent of the optimal
    quality_threshold = optimal_length * (1 + quality_percent / 100)
    
    # Reliability: percentage of runs that achieved quality threshold
    runs_within_quality = sum(1 for length in best_lengths if length <= quality_threshold)
    reliability_percent = (runs_within_quality / len(best_lengths)) * 100
    
    # Speed: average evaluations to reach quality threshold
    evals_within_quality = [evaluations_to_best[i] for i, length in enumerate(best_lengths) 
                           if length <= quality_threshold]
    avg_speed = np.mean(evals_within_quality) if evals_within_quality else 0
    
    # Print results
    print(f"\nBest Tour Lengths Across Runs:")
    print(f"  Average: {avg_best_length:.2f}")
    print(f"  Std Dev: {std_best_length:.2f}")
    print(f"  Min: {min_best_length}")
    print(f"  Max: {max_best_length}")
    
    print(f"\nMetrics:")
    print(f"  Quality: {quality_percent:.2f}% from optimal")
    print(f"    (Avg best tour: {avg_best_length:.2f}, Optimal: {optimal_length})")
    
    print(f"\n  Reliability: {reliability_percent:.2f}%")
    print(f"    ({runs_within_quality}/{len(best_lengths)} runs within quality threshold)")
    
    print(f"\n  Speed: {avg_speed:.0f} evaluations")
    print(f"    (Avg evaluations to reach quality threshold)")
    
    # Analyze 2-opt results if available
    if twoopt_files:
        print(f"\n{'='*60}")
        print("2-OPT LOCAL SEARCH RESULTS (Subpart 2)")
        print(f"{'='*60}")
        
        twoopt_improvements = []
        twoopt_percent_improvements = []
        twoopt_final_lengths = []
        twoopt_num_swaps = []
        
        for twoopt_file in sorted(twoopt_files):
            result = parse_2opt_results_file(twoopt_file)
            if result:
                ga_len, twoopt_len, improvement, percent_imp, swaps = result
                twoopt_improvements.append(improvement)
                twoopt_percent_improvements.append(percent_imp)
                twoopt_final_lengths.append(twoopt_len)
                twoopt_num_swaps.append(swaps)
        
        if twoopt_improvements:
            avg_improvement = np.mean(twoopt_improvements)
            avg_percent_improvement = np.mean(twoopt_percent_improvements)
            avg_final_length = np.mean(twoopt_final_lengths)
            avg_swaps = np.mean(twoopt_num_swaps)
            
            twoopt_quality = ((avg_final_length - optimal_length) / optimal_length) * 100
            
            print(f"\n2-Opt Improvement Statistics:")
            print(f"  Average improvement: {avg_improvement:.2f} units")
            print(f"  Average percent improvement over GA: {avg_percent_improvement:.2f}%")
            print(f"  Average number of 2-opt swaps: {avg_swaps:.0f}")
            
            print(f"\n2-Opt Final Results:")
            print(f"  Average final tour length: {avg_final_length:.2f}")
            print(f"  2-Opt Quality: {twoopt_quality:.2f}% from optimal")
            
            print(f"\nComparison:")
            print(f"  GA Quality: {quality_percent:.2f}% from optimal")
            print(f"  2-Opt Quality: {twoopt_quality:.2f}% from optimal")
            print(f"  Additional improvement by 2-Opt: {quality_percent - twoopt_quality:.2f} percentage points")
    
    # Compute avg-avg and max-avg statistics per generation
    if stats_files:
        print(f"\nComputing aggregate statistics across all runs...")
        
        # Parse all stats files
        all_stats = []
        for stats_file in sorted(stats_files):
            stats = parse_stats_file(stats_file)
            all_stats.append(stats)
        
        # Compute averages per generation
        max_generations = max(len(stats) for stats in all_stats)
        
        avg_best_lengths_per_gen = []
        avg_avg_lengths_per_gen = []
        
        for gen in range(max_generations):
            best_lengths_this_gen = []
            avg_lengths_this_gen = []
            
            for stats in all_stats:
                if gen < len(stats):
                    best_lengths_this_gen.append(stats[gen]['best_length'])
                    avg_lengths_this_gen.append(stats[gen]['avg_length'])
            
            if best_lengths_this_gen:
                avg_best_lengths_per_gen.append(np.mean(best_lengths_this_gen))
                avg_avg_lengths_per_gen.append(np.mean(avg_lengths_this_gen))
        
        # Save aggregate statistics
        output_file = os.path.join(directory, f'{benchmark_name}_aggregate_stats.csv')
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Generation', 'AvgBestLength', 'AvgAvgLength'])
            for gen, (avg_best, avg_avg) in enumerate(zip(avg_best_lengths_per_gen, avg_avg_lengths_per_gen)):
                writer.writerow([gen, f'{avg_best:.2f}', f'{avg_avg:.2f}'])
        
        print(f"Aggregate statistics saved to: {output_file}")
    
    # Prepare return dictionary
    result_dict = {
        'benchmark': benchmark_name,
        'optimal_length': optimal_length,
        'num_runs': len(best_lengths),
        'avg_best_length': avg_best_length,
        'quality_percent': quality_percent,
        'reliability_percent': reliability_percent,
        'speed_evaluations': avg_speed
    }
    
    # Add 2-opt results if available
    if twoopt_files and twoopt_improvements:
        result_dict['twoopt_avg_improvement'] = avg_improvement
        result_dict['twoopt_avg_percent_improvement'] = avg_percent_improvement
        result_dict['twoopt_quality'] = twoopt_quality
        result_dict['twoopt_avg_final_length'] = avg_final_length
    
    return result_dict

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_results.py <experiment_directory>")
        print("Example: python3 analyze_results.py data/berlin52_experiments")
        sys.exit(1)
    
    directory = sys.argv[1]
    
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
        sys.exit(1)
    
    results = analyze_experiment_directory(directory)
    
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"GA Quality: {results['quality_percent']:.2f}%")
        print(f"GA Reliability: {results['reliability_percent']:.2f}%")
        print(f"GA Speed: {results['speed_evaluations']:.0f} evaluations")
        
        if 'twoopt_quality' in results:
            print(f"\n2-Opt Quality: {results['twoopt_quality']:.2f}%")
            print(f"2-Opt Improvement: {results['twoopt_avg_percent_improvement']:.2f}% over GA")
        
        print(f"{'='*60}")

if __name__ == '__main__':
    main()