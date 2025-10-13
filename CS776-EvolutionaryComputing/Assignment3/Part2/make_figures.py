import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def read_ga_run(filename):
    """Read a GA run file and return evaluations, max_fitness, avg_fitness."""
    evals, max_fit, avg_fit = [], [], []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) == 3:
                evals.append(int(parts[0]))
                max_fit.append(float(parts[1]))
                avg_fit.append(float(parts[2]))
    return np.array(evals), np.array(max_fit), np.array(avg_fit)

def read_hc_run(filename):
    """Read a HC run file and return evaluations, fitness."""
    evals, fitness = [], []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                evals.append(int(parts[0]))
                fitness.append(float(parts[1]))
    return np.array(evals), np.array(fitness)

def compute_ga_averages(data_dir, num_runs=30):
    """Compute average max and average avg fitness across all GA runs."""
    all_max_fitness = []
    all_avg_fitness = []
    evals = None
    
    for run in range(num_runs):
        filename = Path(data_dir) / f'ga_run_{run}.txt'
        if filename.exists():
            e, max_fit, avg_fit = read_ga_run(filename)
            if evals is None:
                evals = e
            all_max_fitness.append(max_fit)
            all_avg_fitness.append(avg_fit)
    
    # Compute averages across runs
    avg_max = np.mean(all_max_fitness, axis=0)
    avg_avg = np.mean(all_avg_fitness, axis=0)
    
    return evals, avg_max, avg_avg

def compute_hc_averages(data_dir, num_runs=30):
    """Compute average fitness across all HC runs."""
    all_fitness = []
    evals = None
    
    for run in range(num_runs):
        filename = Path(data_dir) / f'hc_run_{run}.txt'
        if filename.exists():
            e, fitness = read_hc_run(filename)
            if evals is None:
                evals = e
            all_fitness.append(fitness)
    
    # Compute average across runs
    avg_fitness = np.mean(all_fitness, axis=0)
    
    return evals, avg_fitness

def plot_fitness_curves(data_dir='data', output_file='ga_vs_hc_fitness.png'):
    """Create the fitness curves plot comparing GA and HC."""
    
    # Read and process data
    ga_evals, ga_avg_max, ga_avg_avg = compute_ga_averages(data_dir)
    hc_evals, hc_avg_fitness = compute_hc_averages(data_dir)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot GA curves
    plt.plot(ga_evals, ga_avg_max, 'b-', linewidth=2, label='GA Avg-Max', marker='o', 
             markevery=len(ga_evals)//10, markersize=6)
    plt.plot(ga_evals, ga_avg_avg, 'b--', linewidth=2, label='GA Avg-Avg', marker='s', 
             markevery=len(ga_evals)//10, markersize=6, alpha=0.7)
    
    # Plot HC curves
    # For HC, we only have one fitness track (current best), so we plot it twice with same data
    plt.plot(hc_evals, hc_avg_fitness, 'r-', linewidth=2, label='HC Avg-Max', marker='^', 
             markevery=len(hc_evals)//5, markersize=6)
    plt.plot(hc_evals, hc_avg_fitness, 'r--', linewidth=2, label='HC Avg-Avg', marker='v', 
             markevery=len(hc_evals)//5, markersize=6, alpha=0.7)
    
    # Formatting
    plt.xlabel('Number of Evaluations', fontsize=12, fontweight='bold')
    plt.ylabel('Fitness', fontsize=12, fontweight='bold')
    plt.title('GA vs. Hill Climber - Fitness Curves (30 Runs)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'Figure saved to {output_file}')
    
    # Display statistics
    print(f'\nGA Final Statistics:')
    print(f'  Avg-Max: {ga_avg_max[-1]:.2f}')
    print(f'  Avg-Avg: {ga_avg_avg[-1]:.2f}')
    print(f'\nHC Final Statistics:')
    print(f'  Avg Fitness: {hc_avg_fitness[-1]:.2f}')
    
    plt.show()

if __name__ == '__main__':
    # Run the plotting function
    plot_fitness_curves(data_dir='data', output_file='ga_vs_hc_fitness.png')