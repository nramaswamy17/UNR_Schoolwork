import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
import numpy as np
from collections import defaultdict

# Global configuration
NUM_RUNS = 30  # Number of runs to expect for each function

def analyze_solutions():
    """Analyze candidate solutions from CSV files"""
    # Define the data directory
    data_dir = "data/"
    
    # Get all CSV files matching the patterns
    csv_files = glob.glob(os.path.join(data_dir, "*_data_run*.csv"))
    csv_files.extend(glob.glob(os.path.join(data_dir, "bb*_run_*.csv")))
    
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
        # Handle both patterns: FUNCTION_data_runNUMBER.csv and bbNUMBER_run_NUMBER.csv
        match = re.match(r'(.+)_data_run(\d+)\.csv', filename) or re.match(r'(bb\d+)_run_(\d+)\.csv', filename)
        
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
    functions = ['onemax', 'deceptive_trap', 'bb1', 'bb2']
    
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
            
            # Extract solution bit string (bit0, bit1, bit2, etc.) if available
            solution_cols = [col for col in df.columns if col.startswith('bit')]
            if solution_cols:
                best_solution = best_row[solution_cols].values
            else:
                # For black box data, we don't have solution vectors
                best_solution = np.array([])
            
            best_solutions_per_run.append(best_solution)
            
            # Check if this is the overall best
            if best_fitness < best_fitness_overall:
                best_fitness_overall = best_fitness
                best_solution_overall = best_solution
                best_run_number = run_number
        
        # 1. Best solution and its fitness score
        print(f"1. Best Solution (Run {best_run_number}):")
        print(f"   Fitness: {best_fitness_overall:.6f}")
        if len(best_solution_overall) > 0:
            print(f"   Solution: {best_solution_overall}")
        else:
            print(f"   Solution: [Black box - solution vector not stored]")
        
        # 2. Average best solution
        if best_solutions_per_run and len(best_solutions_per_run[0]) > 0:
            best_solutions_array = np.array(best_solutions_per_run)
            average_solution = np.mean(best_solutions_array, axis=0)
            print(f"\n2. Average Best Solution:")
            print(f"   Solution: {average_solution}")
            
            # 3. Standard deviation of solution dimensions
            std_dev_solution = np.std(best_solutions_array, axis=0)
            print(f"\n3. Standard Deviation of Solution Dimensions:")
            for i, std in enumerate(std_dev_solution):
                print(f"   bit{i}: {std:.6f}")
        else:
            print(f"\n2. Average Best Solution:")
            print(f"   Solution: [Black box - solution vectors not available]")
            print(f"\n3. Standard Deviation of Solution Dimensions:")
            print(f"   [Not applicable for black box data]")
        
        # Additional statistics
        print(f"\nAdditional Statistics:")
        print(f"   Number of runs: {len(runs_data)} (expected: {NUM_RUNS})")
        print(f"   Solution dimensions: {len(solution_cols)}")
        print(f"   Average fitness across runs: {np.mean([df.iloc[-1]['Fitness'] for df in [run['data'] for run in runs_data]]):.6f}")
        print(f"   Std dev of fitness across runs: {np.std([df.iloc[-1]['Fitness'] for df in [run['data'] for run in runs_data]]):.6f}")
        
        # Validate number of runs
        if len(runs_data) != NUM_RUNS:
            print(f"   WARNING: Expected {NUM_RUNS} runs but found {len(runs_data)}")

def read_and_plot_data():
    # Define the data directory
    data_dir = "data/"
    
    # Get all CSV files matching the patterns
    csv_files = glob.glob(os.path.join(data_dir, "*_data_run*.csv"))
    csv_files.extend(glob.glob(os.path.join(data_dir, "bb*_run_*.csv")))
    
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
        # Handle both patterns: FUNCTION_data_runNUMBER.csv and bbNUMBER_run_NUMBER.csv
        match = re.match(r'(.+)_data_run(\d+)\.csv', filename) or re.match(r'(bb\d+)_run_(\d+)\.csv', filename)
        
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
    functions = ['onemax', 'deceptive_trap', 'bb1', 'bb2']
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Fitness vs Iteration for Different Functions', fontsize=16)
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
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
            
            # Format function name for display
            if function_name == 'deceptive_trap':
                display_name = 'Deceptive Trap'
            elif function_name == 'bb1':
                display_name = 'Black Box 1'
            elif function_name == 'bb2':
                display_name = 'Black Box 2'
            else:
                display_name = function_name.capitalize()
            ax.set_title(f'{display_name} Function')
            ax.set_xlabel('Iteration')
            #ax.grid(True, alpha=0.3)
            
            # Set y-axis label and limits
            ax.set_ylabel('Fitness')
            ax.set_ylim(0, 100)
            
            print(f"Plotted {len(runs_data)} runs for {function_name}")
        else:
            ax.text(0.5, 0.5, f'No data found for\n{function_name}', 
                   transform=ax.transAxes, ha='center', va='center')
            # Format function name for display
            if function_name == 'deceptive_trap':
                display_name = 'Deceptive Trap'
            elif function_name == 'bb1':
                display_name = 'Black Box 1'
            elif function_name == 'bb2':
                display_name = 'Black Box 2'
            else:
                display_name = function_name.capitalize()
            ax.set_title(f'{display_name} Function')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n=== Summary (Expected {NUM_RUNS} runs per function) ===")
    for function_name in functions:
        if function_name in function_data:
            num_runs = len(function_data[function_name])
            status = "✓" if num_runs == NUM_RUNS else "⚠"
            print(f"{status} {function_name}: {num_runs} runs found")
        else:
            print(f"✗ {function_name}: No runs found")

if __name__ == "__main__":
    # Analyze candidate solutions
    analyze_solutions()
    
    # Plot fitness curves
    read_and_plot_data()