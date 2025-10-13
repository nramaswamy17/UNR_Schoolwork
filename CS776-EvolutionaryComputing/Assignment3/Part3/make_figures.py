#!/usr/bin/env python3
"""
Generate TSP GA experiment visualization figures
Creates fitness curves and objective (tour length) curves
"""

import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')

def read_stats_file(filename):
    """Read a stats CSV file and return the data"""
    generations = []
    best_lengths = []
    avg_lengths = []
    evaluations = []
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            generations.append(int(row['Generation']))
            best_lengths.append(int(row['BestLength']))
            avg_lengths.append(float(row['AvgLength']))
            evaluations.append(int(row['Evaluations']))
    
    return {
        'generations': np.array(generations),
        'best_lengths': np.array(best_lengths),
        'avg_lengths': np.array(avg_lengths),
        'evaluations': np.array(evaluations)
    }

def compute_aggregate_stats(experiment_dir, benchmark_name):
    """Compute aggregate statistics across all runs"""
    
    # Find all stats files for this benchmark
    pattern = os.path.join(experiment_dir, f'{benchmark_name}_stats_seed*.csv')
    stats_files = sorted(glob.glob(pattern))
    
    if not stats_files:
        print(f"No stats files found for {benchmark_name} in {experiment_dir}")
        return None
    
    print(f"Found {len(stats_files)} runs for {benchmark_name}")
    
    # Read all stats files
    all_stats = []
    for stats_file in stats_files:
        try:
            stats = read_stats_file(stats_file)
            all_stats.append(stats)
        except Exception as e:
            print(f"Error reading {stats_file}: {e}")
            continue
    
    if not all_stats:
        return None
    
    # Determine the maximum number of generations
    max_generations = max(len(stats['generations']) for stats in all_stats)
    
    # Initialize arrays for aggregate statistics
    avg_best_lengths = []
    avg_avg_lengths = []
    avg_best_fitness = []
    avg_avg_fitness = []
    generations_axis = []
    
    # Compute averages for each generation
    for gen in range(max_generations):
        best_lengths_this_gen = []
        avg_lengths_this_gen = []
        
        for stats in all_stats:
            if gen < len(stats['generations']):
                best_lengths_this_gen.append(stats['best_lengths'][gen])
                avg_lengths_this_gen.append(stats['avg_lengths'][gen])
        
        if best_lengths_this_gen:
            generations_axis.append(gen)
            
            # Average best length across runs
            avg_best_length = np.mean(best_lengths_this_gen)
            avg_best_lengths.append(avg_best_length)
            
            # Average avg length across runs
            avg_avg_length = np.mean(avg_lengths_this_gen)
            avg_avg_lengths.append(avg_avg_length)
            
            # Convert to fitness (1/length)
            avg_best_fitness.append(1.0 / avg_best_length)
            avg_avg_fitness.append(1.0 / avg_avg_length)
    
    return {
        'generations': np.array(generations_axis),
        'avg_best_lengths': np.array(avg_best_lengths),
        'avg_avg_lengths': np.array(avg_avg_lengths),
        'avg_best_fitness': np.array(avg_best_fitness),
        'avg_avg_fitness': np.array(avg_avg_fitness),
        'num_runs': len(all_stats)
    }

def create_fitness_curves(aggregate_stats, benchmark_name, output_dir='figures'):
    """Create fitness curves plot"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    generations = aggregate_stats['generations']
    avg_best_fitness = aggregate_stats['avg_best_fitness']
    avg_avg_fitness = aggregate_stats['avg_avg_fitness']
    
    # Plot the curves
    ax.plot(generations, avg_best_fitness, 'b-', linewidth=2, 
            label='Max-Avg Fitness (Avg of Best)', marker='o', markersize=3, markevery=max(1, len(generations)//20))
    ax.plot(generations, avg_avg_fitness, 'r-', linewidth=2, 
            label='Avg-Avg Fitness (Avg of Avg)', marker='s', markersize=3, markevery=max(1, len(generations)//20))
    
    ax.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fitness (1/Tour Length)', fontsize=12, fontweight='bold')
    ax.set_title(f'TSP Fitness Curves - {benchmark_name.upper()}\n({aggregate_stats["num_runs"]} runs)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'{benchmark_name}_fitness_curves.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved fitness curves to {output_file}")
    plt.close()

def create_objective_curves(aggregate_stats, benchmark_name, optimal_length=None, output_dir='figures'):
    """Create objective (tour length) curves plot"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    generations = aggregate_stats['generations']
    avg_best_lengths = aggregate_stats['avg_best_lengths']
    avg_avg_lengths = aggregate_stats['avg_avg_lengths']
    
    # Plot the curves
    ax.plot(generations, avg_best_lengths, 'b-', linewidth=2, 
            label='Avg-Best Tour Length', marker='o', markersize=3, markevery=max(1, len(generations)//20))
    ax.plot(generations, avg_avg_lengths, 'r-', linewidth=2, 
            label='Avg-Avg Tour Length', marker='s', markersize=3, markevery=max(1, len(generations)//20))
    
    # Add optimal tour length line if provided
    if optimal_length:
        ax.axhline(y=optimal_length, color='g', linestyle='--', linewidth=2, 
                   label=f'Optimal Tour Length ({optimal_length})')
    
    ax.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tour Length (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title(f'TSP Objective Curves - {benchmark_name.upper()}\n({aggregate_stats["num_runs"]} runs)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'{benchmark_name}_objective_curves.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved objective curves to {output_file}")
    plt.close()

def main():
    """Main function to generate all figures"""
    
    # Optimal tour lengths for benchmarks
    OPTIMAL_LENGTHS = {
        'burma14': 3323,
        'berlin52': 7542,
        'eil51': 426,
        'eil76': 538,
        'lin105': 14379,
        'lin318': 42029
    }
    
    # Base data directory
    base_data_dir = 'data'
    output_dir = 'figures'
    
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
    print("=" * 60)
    
    # Process each benchmark
    for benchmark_name, exp_dir in experiment_dirs:
        print(f"\nProcessing {benchmark_name}...")
        print("-" * 60)
        
        # Compute aggregate statistics
        aggregate_stats = compute_aggregate_stats(exp_dir, benchmark_name)
        
        if aggregate_stats is None:
            print(f"Skipping {benchmark_name} - no valid data found")
            continue
        
        # Create fitness curves
        create_fitness_curves(aggregate_stats, benchmark_name, output_dir)
        
        # Create objective curves
        optimal_length = OPTIMAL_LENGTHS.get(benchmark_name)
        create_objective_curves(aggregate_stats, benchmark_name, optimal_length, output_dir)
        
        print(f"Completed {benchmark_name}")
    
    print("\n" + "=" * 60)
    print(f"All figures saved to '{output_dir}/' directory")
    print("=" * 60)

if __name__ == '__main__':
    main()