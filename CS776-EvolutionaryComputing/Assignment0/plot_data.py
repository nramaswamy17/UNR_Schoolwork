import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
import numpy as np
from collections import defaultdict

def analyze_solutions():
    """Analyze candidate solutions from CSV files"""
    # Define the data directory
    data_dir = "data/"
    
    # Get all CSV files matching the pattern
    csv_files = glob.glob(os.path.join(data_dir, "*_data_run*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    # Dictionary to store data for each function
    function_data = defaultdict(list)
    
    # Process each file
    for file_path in csv_files:
        # Extract filename without path
        filename = os.path.basename(file_path)
        
        # Parse the filename to extract function name and run number
        match = re.match(r'([^_]+)_data_run(\d+)\.csv', filename)
        
        if match:
            function_name = match.group(1)
            run_number = int(match.group(2))
            
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Store the data with function name as key
                function_data[function_name].append({
                    'run_number': run_number,
                    'data': df
                })
                
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        else:
            print(f"Filename {filename} doesn't match expected pattern")
    
    # Analyze each function
    functions = ['sphere', 'rosenbrock', 'step']
    
    for function_name in functions:
        if function_name not in function_data:
            print(f"\n=== {function_name.upper()} FUNCTION ===")
            print("No data found")
            continue
            
        print(f"\n=== {function_name.upper()} FUNCTION ===")
        runs_data = function_data[function_name]
        
        # Find the best solution across all runs
        best_solution_overall = None
        best_fitness_overall = float('inf')
        best_run_number = None
        
        # Collect best solutions from each run
        best_solutions_per_run = []
        
        for run_info in runs_data:
            df = run_info['data']
            run_number = run_info['run_number']
            
            # Get the best solution from this run (last row has the best fitness)
            best_row = df.iloc[-1]
            best_fitness = best_row['Fitness']
            
            # Extract solution coordinates (x0, x1, x2, etc.)
            solution_cols = [col for col in df.columns if col.startswith('x')]
            best_solution = best_row[solution_cols].values
            
            best_solutions_per_run.append(best_solution)
            
            # Check if this is the overall best
            if best_fitness < best_fitness_overall:
                best_fitness_overall = best_fitness
                best_solution_overall = best_solution
                best_run_number = run_number
        
        # 1. Best solution and its fitness score
        print(f"1. Best Solution (Run {best_run_number}):")
        print(f"   Fitness: {best_fitness_overall:.6f}")
        print(f"   Solution: {best_solution_overall}")
        
        # 2. Average best solution
        best_solutions_array = np.array(best_solutions_per_run)
        average_solution = np.mean(best_solutions_array, axis=0)
        print(f"\n2. Average Best Solution:")
        print(f"   Solution: {average_solution}")
        
        # 3. Standard deviation of solution dimensions
        std_dev_solution = np.std(best_solutions_array, axis=0)
        print(f"\n3. Standard Deviation of Solution Dimensions:")
        for i, std in enumerate(std_dev_solution):
            print(f"   x{i}: {std:.6f}")
        
        # Additional statistics
        print(f"\nAdditional Statistics:")
        print(f"   Number of runs: {len(runs_data)}")
        print(f"   Solution dimensions: {len(solution_cols)}")
        print(f"   Average fitness across runs: {np.mean([df.iloc[-1]['Fitness'] for df in [run['data'] for run in runs_data]]):.6f}")
        print(f"   Std dev of fitness across runs: {np.std([df.iloc[-1]['Fitness'] for df in [run['data'] for run in runs_data]]):.6f}")

def read_and_plot_data():
    # Define the data directory
    data_dir = "data/"
    
    # Get all CSV files matching the pattern
    csv_files = glob.glob(os.path.join(data_dir, "*_data_run*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    # Dictionary to store data for each function
    function_data = defaultdict(list)
    
    # Process each file
    for file_path in csv_files:
        # Extract filename without path
        filename = os.path.basename(file_path)
        
        # Parse the filename to extract function name and run number
        # Pattern: FUNCTION_data_runNUMBER.csv
        match = re.match(r'([^_]+)_data_run(\d+)\.csv', filename)
        
        if match:
            function_name = match.group(1)
            run_number = int(match.group(2))
            
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Store the data with function name as key
                function_data[function_name].append({
                    'run_number': run_number,
                    'data': df
                })
                
                #print(f"Loaded {filename}: {function_name}, run {run_number}")
                
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        else:
            print(f"Filename {filename} doesn't match expected pattern")
    
    # Create plots for each function
    functions = ['sphere', 'rosenbrock', 'step']
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Fitness vs Iteration for Different Functions', fontsize=16)
    
    for i, function_name in enumerate(functions):
        ax = axes[i]
        
        if function_name in function_data:
            runs_data = function_data[function_name]
            
            # Sort by run number for consistent plotting
            runs_data.sort(key=lambda x: x['run_number'])
            
            # Find the run with the lowest final fitness value
            best_run = None
            best_final_fitness = float('inf')
            
            for run_info in runs_data:
                df = run_info['data']
                final_fitness = df.iloc[-1]['Fitness']
                if final_fitness < best_final_fitness:
                    best_final_fitness = final_fitness
                    best_run = run_info
            
            # Plot each run
            for run_info in runs_data:
                run_num = run_info['run_number']
                df = run_info['data']
                
                # Check if this is the best run
                if run_info == best_run:
                    # Plot best run in bold green, right behind the red line
                    ax.plot(df['Iteration'], df['Fitness'], 
                           color='green', alpha=0.9, linewidth=3, label=f'Run {run_num} (Best)', zorder=2)
                else:
                    # Plot other runs in light gray
                    ax.plot(df['Iteration'], df['Fitness'], 
                           color='lightgray', alpha=0.7, linewidth=0.8, label=f'Run {run_num}')
            
            # Calculate and plot the average line
            if runs_data:
                # Get the maximum number of iterations across all runs
                max_iterations = max(len(run_info['data']) for run_info in runs_data)
                
                # Calculate average fitness for each iteration
                avg_fitness = []
                for iteration in range(max_iterations):
                    fitness_values = []
                    for run_info in runs_data:
                        df = run_info['data']
                        if iteration < len(df):
                            fitness_values.append(df.iloc[iteration]['Fitness'])
                    
                    if fitness_values:
                        avg_fitness.append(sum(fitness_values) / len(fitness_values))
                
                # Plot the average line in bold red
                iterations = list(range(len(avg_fitness)))
                ax.plot(iterations, avg_fitness, 
                       color='red', linewidth=3, label='Average', alpha=0.9, zorder=3)
            
            ax.set_title(f'{function_name.capitalize()} Function')
            ax.set_xlabel('Iteration')
            #ax.grid(True, alpha=0.3)
            
            # Set y-axis to log scale if fitness values vary greatly
            if function_name == 'rosenbrock' or function_name == 'sphere':
                ax.set_yscale('log')
                ax.set_ylabel('log(Fitness)')
            else:
                ax.set_ylabel('Fitness')
            
            print(f"Plotted {len(runs_data)} runs for {function_name}")
        else:
            ax.text(0.5, 0.5, f'No data found for\n{function_name}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{function_name.capitalize()} Function')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n=== Summary ===")
    for function_name in functions:
        if function_name in function_data:
            num_runs = len(function_data[function_name])
            print(f"{function_name}: {num_runs} runs found")
        else:
            print(f"{function_name}: No runs found")

if __name__ == "__main__":
    # Analyze candidate solutions
    analyze_solutions()
    
    # Plot fitness curves
    read_and_plot_data()