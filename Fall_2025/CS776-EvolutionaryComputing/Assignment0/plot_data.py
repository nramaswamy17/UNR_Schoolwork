import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
from collections import defaultdict

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
                
                print(f"Loaded {filename}: {function_name}, run {run_number}")
                
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
            
            # Plot each run in light gray
            for run_info in runs_data:
                run_num = run_info['run_number']
                df = run_info['data']
                
                # Plot with light gray color and thin lines
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
                       color='red', linewidth=3, label='Average', alpha=0.9)
            
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
    read_and_plot_data()