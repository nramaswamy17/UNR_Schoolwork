import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import numpy as np

# Find all CSV results (excluding summaries)
csv_files = glob.glob('results/*.csv')

if not csv_files:
    print("No results found!")
    exit()

print(f"Found {len(csv_files)} result file(s)")

# Process each instance file
for filepath in csv_files:
    df = pd.read_csv(filepath)
    basename = os.path.basename(filepath).replace('.csv', '')
    
    print(f"\nProcessing {basename}...")
    
    # Get unique seeds
    seeds = df['seed'].unique()
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    
    all_data = []
    for seed in seeds:
        seed_data = df[df['seed'] == seed]
        plt.plot(seed_data['generation'], seed_data['objective'], 
                alpha=0.7, label=f'Seed {seed}')
        all_data.append(seed_data['objective'].values)
    
    # Plot mean across seeds
    if len(all_data) > 1:
        mean_obj = np.mean(all_data, axis=0)
        plt.plot(range(len(mean_obj)), mean_obj, 'k--', linewidth=2, label='Mean')
    
    plt.xlabel('Generation')
    plt.ylabel('Objective (Min-Max Tour Length)')
    plt.title(f'{basename} - Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = f'results/{basename}_convergence.png'
    plt.savefig(output_file, dpi=150)
    print(f"  Saved: {output_file}")
    plt.close()
    
    # Print statistics
    print(f"  Statistics:")
    best_objs = []
    for seed in seeds:
        seed_data = df[df['seed'] == seed]
        best_obj = seed_data['objective'].min()
        best_objs.append(best_obj)
        print(f"    Seed {seed}: {best_obj:.2f}")
    
    if len(best_objs) > 1:
        print(f"    Mean: {np.mean(best_objs):.2f}")
        print(f"    Std: {np.std(best_objs):.2f}")
    print(f"    Best: {min(best_objs):.2f}")

print("\nVisualization complete!")