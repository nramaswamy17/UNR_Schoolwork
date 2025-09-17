import numpy as np
import random
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Callable
import json

# Set up data directory
os.makedirs('data', exist_ok=True)

class DeJongFunctions:
    """Implementation of the four DeJong test functions"""
    
    @staticmethod
    def sphere(x: np.ndarray) -> float:
        """F1: Sphere function, domain [-5.12, 5.12], min at origin = 0"""
        return np.sum(x**2)
    
    @staticmethod
    def rosenbrock(x: np.ndarray) -> float:
        """F2: Rosenbrock function, domain [-2.048, 2.048], min at (1,1) = 0"""
        total = 0.0
        for i in range(len(x) - 1):
            total += 100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
        return total
    
    @staticmethod
    def step(x: np.ndarray) -> float:
        """F3: Step function, domain [-5.12, 5.12], min at (-5.12,...) = -5.12*n"""
        return np.sum(np.floor(x))
    
    @staticmethod
    def quartic_with_noise(x: np.ndarray) -> float:
        """F4: Quartic with noise, domain [-1.28, 1.28], min at origin ≈ 0"""
        total = 0.0
        for i, xi in enumerate(x):
            total += (i + 1) * xi**4
        # Add Gaussian noise
        noise = np.random.normal(0, 1)
        return total + noise

class BinaryEncoder:
    """Handles binary encoding/decoding of real values"""
    
    def __init__(self, bits_per_var: int = 16):
        self.bits_per_var = bits_per_var
        self.max_int = (2**bits_per_var) - 1
    
    def encode(self, real_values: np.ndarray, domains: List[Tuple[float, float]]) -> np.ndarray:
        """Convert real values to binary representation"""
        binary = []
        for val, (min_val, max_val) in zip(real_values, domains):
            # Normalize to [0, 1]
            normalized = (val - min_val) / (max_val - min_val)
            # Convert to integer
            int_val = int(normalized * self.max_int)
            int_val = max(0, min(int_val, self.max_int))  # Clamp
            # Convert to binary
            binary_str = format(int_val, f'0{self.bits_per_var}b')
            binary.extend([int(b) for b in binary_str])
        return np.array(binary)
    
    def decode(self, binary: np.ndarray, domains: List[Tuple[float, float]]) -> np.ndarray:
        """Convert binary representation to real values"""
        real_values = []
        for i, (min_val, max_val) in enumerate(domains):
            start_idx = i * self.bits_per_var
            end_idx = start_idx + self.bits_per_var
            binary_chunk = binary[start_idx:end_idx]
            
            # Convert binary to integer
            int_val = 0
            for bit in binary_chunk:
                int_val = (int_val << 1) + bit
            
            # Normalize and scale
            normalized = int_val / self.max_int
            real_val = min_val + normalized * (max_val - min_val)
            real_values.append(real_val)
        
        return np.array(real_values)

class GeneticAlgorithm:
    """Base class for genetic algorithms"""
    
    def __init__(self, pop_size: int, num_generations: int, function: Callable, 
                 domains: List[Tuple[float, float]], seed: int = None):
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.function = function
        self.domains = domains
        self.encoder = BinaryEncoder()
        self.chromosome_length = len(domains) * self.encoder.bits_per_var
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Statistics tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.worst_fitness_history = []
        self.best_objective_history = []
        self.avg_objective_history = []
        self.worst_objective_history = []
    
    def fitness(self, objective_value: float) -> float:
        """Convert minimization objective to maximization fitness"""
        # Handle edge case where objective_value is -1 (would cause division by zero)
        if objective_value <= -1.0:
            # For very negative objective values, use a large but finite fitness
            return 1e6
        return 1.0 / (1.0 + objective_value)
    
    def evaluate_population(self, population: List[np.ndarray]) -> List[float]:
        """Evaluate fitness of entire population"""
        fitnesses = []
        objectives = []
        
        for individual in population:
            real_values = self.encoder.decode(individual, self.domains)
            obj_val = self.function(real_values)
            fit_val = self.fitness(obj_val)
            fitnesses.append(fit_val)
            objectives.append(obj_val)
        
        return fitnesses, objectives
    
    def record_statistics(self, fitnesses: List[float], objectives: List[float]):
        """Record generation statistics"""
        self.best_fitness_history.append(max(fitnesses))
        self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
        self.worst_fitness_history.append(min(fitnesses))
        
        self.best_objective_history.append(min(objectives))
        self.avg_objective_history.append(sum(objectives) / len(objectives))
        self.worst_objective_history.append(max(objectives))

class SimpleGA(GeneticAlgorithm):
    """Simple Genetic Algorithm implementation"""
    
    def __init__(self, pop_size: int = 50, num_generations: int = 100, 
                 crossover_prob: float = 0.7, mutation_prob: float = 0.001,
                 function: Callable = None, domains: List[Tuple[float, float]] = None,
                 seed: int = None):
        super().__init__(pop_size, num_generations, function, domains, seed)
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
    
    def initialize_population(self) -> List[np.ndarray]:
        """Create random initial population"""
        population = []
        for _ in range(self.pop_size):
            individual = np.random.randint(0, 2, self.chromosome_length)
            population.append(individual)
        return population
    
    def fitness_proportional_selection(self, population: List[np.ndarray], 
                                     fitnesses: List[float]) -> np.ndarray:
        """Select individual using fitness proportional selection"""
        # Handle edge cases with invalid fitness values
        fitnesses = [f if np.isfinite(f) and f >= 0 else 1e-6 for f in fitnesses]
        
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            return random.choice(population)
        
        selection_prob = [f / total_fitness for f in fitnesses]
        
        # Ensure probabilities are valid (no NaN or negative values)
        selection_prob = [p if np.isfinite(p) and p >= 0 else 1e-6 for p in selection_prob]
        
        # Normalize probabilities to sum to 1
        prob_sum = sum(selection_prob)
        if prob_sum > 0:
            selection_prob = [p / prob_sum for p in selection_prob]
        else:
            # Fallback to uniform selection
            selection_prob = [1.0 / len(population)] * len(population)
        
        selected_idx = np.random.choice(len(population), p=selection_prob)
        return population[selected_idx].copy()
    
    def one_point_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform one-point crossover"""
        if random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        
        return child1, child2
    
    def bit_flip_mutation(self, individual: np.ndarray) -> np.ndarray:
        """Apply bit flip mutation"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_prob:
                mutated[i] = 1 - mutated[i]
        return mutated
    
    def run(self) -> dict:
        """Run the Simple GA"""
        population = self.initialize_population()
        
        for generation in range(self.num_generations):
            # Evaluate population
            fitnesses, objectives = self.evaluate_population(population)
            self.record_statistics(fitnesses, objectives)
            
            # Create new population
            new_population = []
            
            while len(new_population) < self.pop_size:
                # Selection
                parent1 = self.fitness_proportional_selection(population, fitnesses)
                parent2 = self.fitness_proportional_selection(population, fitnesses)
                
                # Crossover
                child1, child2 = self.one_point_crossover(parent1, parent2)
                
                # Mutation
                child1 = self.bit_flip_mutation(child1)
                child2 = self.bit_flip_mutation(child2)
                
                new_population.extend([child1, child2])
            
            # Keep only pop_size individuals
            population = new_population[:self.pop_size]
        
        # Final evaluation
        fitnesses, objectives = self.evaluate_population(population)
        self.record_statistics(fitnesses, objectives)
        
        return {
            'best_fitness': self.best_fitness_history,
            'avg_fitness': self.avg_fitness_history,
            'worst_fitness': self.worst_fitness_history,
            'best_objective': self.best_objective_history,
            'avg_objective': self.avg_objective_history,
            'worst_objective': self.worst_objective_history,
            'final_best_objective': min(objectives)
        }

class CHCGA(GeneticAlgorithm):
    """CHC Genetic Algorithm implementation"""
    
    def __init__(self, pop_size: int = 50, num_generations: int = 75,
                 crossover_prob: float = 0.95, divergence_rate: float = 0.35,
                 function: Callable = None, domains: List[Tuple[float, float]] = None,
                 seed: int = None):
        super().__init__(pop_size, num_generations, function, domains, seed)
        self.crossover_prob = crossover_prob
        self.divergence_rate = divergence_rate
        self.difference_threshold = self.chromosome_length // 4
        self.restart_count = 0
    
    def initialize_population(self) -> List[np.ndarray]:
        """Create random initial population"""
        population = []
        for _ in range(self.pop_size):
            individual = np.random.randint(0, 2, self.chromosome_length)
            population.append(individual)
        return population
    
    def hamming_distance(self, ind1: np.ndarray, ind2: np.ndarray) -> int:
        """Calculate Hamming distance between two individuals"""
        return np.sum(ind1 != ind2)
    
    def hux_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """HUX crossover: exchange exactly half of differing bits"""
        if random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        # Find differing positions
        diff_positions = np.where(parent1 != parent2)[0]
        
        if len(diff_positions) == 0:
            return parent1.copy(), parent2.copy()
        
        # Select half of differing positions to exchange
        num_to_exchange = len(diff_positions) // 2
        if num_to_exchange == 0:
            return parent1.copy(), parent2.copy()
        
        exchange_positions = np.random.choice(diff_positions, num_to_exchange, replace=False)
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Exchange bits at selected positions
        child1[exchange_positions] = parent2[exchange_positions]
        child2[exchange_positions] = parent1[exchange_positions]
        
        return child1, child2
    
    def incest_prevention(self, population: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Select pairs for mating with incest prevention"""
        shuffled_pop = population.copy()
        random.shuffle(shuffled_pop)
        
        mating_pairs = []
        for i in range(0, len(shuffled_pop) - 1, 2):
            parent1 = shuffled_pop[i]
            parent2 = shuffled_pop[i + 1]
            
            hamming_dist = self.hamming_distance(parent1, parent2)
            if hamming_dist // 2 > self.difference_threshold:
                mating_pairs.append((parent1, parent2))
        
        return mating_pairs
    
    def elitist_selection(self, population: List[np.ndarray], children: List[np.ndarray]) -> List[np.ndarray]:
        """Select best individuals from combined parent and child populations"""
        combined_pop = population + children
        
        if not combined_pop:
            return population
        
        # Evaluate combined population
        fitnesses, _ = self.evaluate_population(combined_pop)
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitnesses)[::-1]
        
        # Select top pop_size individuals
        new_population = [combined_pop[i] for i in sorted_indices[:self.pop_size]]
        
        return new_population
    
    def restart_population(self, best_individual: np.ndarray) -> List[np.ndarray]:
        """Restart population by diverging from best individual"""
        new_population = [best_individual.copy()]
        
        for _ in range(self.pop_size - 1):
            new_individual = best_individual.copy()
            
            # Flip divergence_rate fraction of bits
            num_flips = int(self.divergence_rate * len(new_individual))
            flip_positions = np.random.choice(len(new_individual), num_flips, replace=False)
            new_individual[flip_positions] = 1 - new_individual[flip_positions]
            
            new_population.append(new_individual)
        
        return new_population
    
    def run(self) -> dict:
        """Run the CHC GA"""
        population = self.initialize_population()
        
        for generation in range(self.num_generations):
            # Evaluate population
            fitnesses, objectives = self.evaluate_population(population)
            self.record_statistics(fitnesses, objectives)
            
            # Get mating pairs with incest prevention
            mating_pairs = self.incest_prevention(population)
            
            # Generate children
            children = []
            for parent1, parent2 in mating_pairs:
                child1, child2 = self.hux_crossover(parent1, parent2)
                children.extend([child1, child2])
            
            # Elitist selection
            old_pop_size = len(population)
            population = self.elitist_selection(population, children)
            
            # Check for convergence (no new children accepted)
            if len(children) == 0 or len(population) == old_pop_size:
                self.difference_threshold -= 1
                
                # If threshold drops to zero, restart
                if self.difference_threshold < 0:
                    best_idx = np.argmax(fitnesses)
                    best_individual = population[best_idx]
                    population = self.restart_population(best_individual)
                    self.difference_threshold = int(self.divergence_rate * (1.0 - self.divergence_rate) * self.chromosome_length)
                    self.restart_count += 1
        
        # Final evaluation
        fitnesses, objectives = self.evaluate_population(population)
        self.record_statistics(fitnesses, objectives)
        
        return {
            'best_fitness': self.best_fitness_history,
            'avg_fitness': self.avg_fitness_history,
            'worst_fitness': self.worst_fitness_history,
            'best_objective': self.best_objective_history,
            'avg_objective': self.avg_objective_history,
            'worst_objective': self.worst_objective_history,
            'final_best_objective': min(objectives),
            'restarts': self.restart_count
        }

class HillClimber:
    """Hill Climber implementation for comparison"""
    
    def __init__(self, function: Callable, domains: List[Tuple[float, float]], 
                 max_iterations: int = 1000, seed: int = None):
        self.function = function
        self.domains = domains
        self.max_iterations = max_iterations
        self.encoder = BinaryEncoder()
        self.chromosome_length = len(domains) * self.encoder.bits_per_var
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Statistics tracking
        self.objective_history = []
    
    def run(self) -> dict:
        """Run hill climber with bit-flip mutations"""
        # Random initial solution
        current = np.random.randint(0, 2, self.chromosome_length)
        real_values = self.encoder.decode(current, self.domains)
        current_objective = self.function(real_values)
        
        for iteration in range(self.max_iterations):
            self.objective_history.append(current_objective)
            
            # Generate neighbor by flipping a random bit
            neighbor = current.copy()
            flip_index = random.randint(0, len(neighbor) - 1)
            neighbor[flip_index] = 1 - neighbor[flip_index]
            
            # Evaluate neighbor
            real_values = self.encoder.decode(neighbor, self.domains)
            neighbor_objective = self.function(real_values)
            
            # Accept if better (minimization)
            if neighbor_objective < current_objective:
                current = neighbor
                current_objective = neighbor_objective
        
        return {
            'objective_history': self.objective_history,
            'final_best_objective': current_objective
        }

def run_experiments():
    """Run all experiments on DeJong functions"""
    
    # Define test functions and their properties
    test_functions = {
        'sphere': {
            'function': DeJongFunctions.sphere,
            'domains': [(-5.12, 5.12)] * 3,  # 3D as per previous code
            'name': 'Sphere Function'
        },
        'rosenbrock': {
            'function': DeJongFunctions.rosenbrock,
            'domains': [(-2.048, 2.048)] * 2,  # 2D as per previous code
            'name': 'Rosenbrock Function'
        },
        'step': {
            'function': DeJongFunctions.step,
            'domains': [(-5.12, 5.12)] * 5,  # 5D as per previous code
            'name': 'Step Function'
        },
        'quartic': {
            'function': DeJongFunctions.quartic_with_noise,
            'domains': [(-1.28, 1.28)] * 3,  # 3D for quartic
            'name': 'Quartic with Noise Function'
        }
    }
    
    num_runs = 30
    results = {}
    
    for func_name, func_info in test_functions.items():
        print(f"\nRunning experiments on {func_info['name']}...")
        
        results[func_name] = {
            'sga': [],
            'chc': [],
            'hillclimber': []
        }
        
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}")
            seed = run + 1000  # Ensure reproducible but different seeds
            
            # Simple GA
            sga = SimpleGA(
                pop_size=50,
                num_generations=100,
                crossover_prob=0.7,
                mutation_prob=0.001,
                function=func_info['function'],
                domains=func_info['domains'],
                seed=seed
            )
            sga_result = sga.run()
            results[func_name]['sga'].append(sga_result)
            
            # CHC GA
            chc = CHCGA(
                pop_size=50,
                num_generations=75,
                crossover_prob=0.95,
                divergence_rate=0.35,
                function=func_info['function'],
                domains=func_info['domains'],
                seed=seed
            )
            chc_result = chc.run()
            results[func_name]['chc'].append(chc_result)
            
            # Hill Climber
            hc = HillClimber(
                function=func_info['function'],
                domains=func_info['domains'],
                max_iterations=1000,
                seed=seed
            )
            hc_result = hc.run()
            results[func_name]['hillclimber'].append(hc_result)
        
        # Save results for this function
        with open(f'data/{func_name}_results.json', 'w') as f:
            json.dump(results[func_name], f, indent=2)
    
    return results, test_functions

def create_plots(results: dict, test_functions: dict):
    """Create comparison plots for all algorithms and functions"""
    
    for func_name, func_info in test_functions.items():
        print(f"Creating plots for {func_info['name']}...")
        
        # Create figure with subplots (only top two charts)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"{func_info['name']} - Algorithm Comparison", fontsize=16)
        
        # Get data for each algorithm
        sga_data = results[func_name]['sga']
        chc_data = results[func_name]['chc']
        hc_data = results[func_name]['hillclimber']
        
        # Determine maximum generations for plotting
        max_gens_sga = len(sga_data[0]['best_fitness'])
        max_gens_chc = len(chc_data[0]['best_fitness'])
        max_gens_hc = len(hc_data[0]['objective_history'])
        max_gens = max(max_gens_sga, max_gens_chc, max_gens_hc)
        
        # Calculate averages across runs
        def calculate_averages(data, key, convert_to_fitness=False):
            if not data:
                return np.zeros(max_gens)
            
            # Handle different lengths by padding
            all_series = []
            for run_data in data:
                if key in run_data:
                    series = run_data[key]
                    if convert_to_fitness:
                        # Convert objectives to fitness for hill climber
                        series = [1.0 / (1.0 + obj) for obj in series]
                    
                    # Pad series to max_gens length
                    if len(series) < max_gens:
                        # Repeat last value
                        series = series + [series[-1]] * (max_gens - len(series))
                    elif len(series) > max_gens:
                        series = series[:max_gens]
                    
                    all_series.append(series)
            
            if not all_series:
                return np.zeros(max_gens)
            
            return np.mean(all_series, axis=0)
        
        # Calculate averages for each algorithm
        generations = range(max_gens)
        
        # SGA averages
        sga_best_fit = calculate_averages(sga_data, 'best_fitness')
        sga_avg_fit = calculate_averages(sga_data, 'avg_fitness')
        sga_worst_fit = calculate_averages(sga_data, 'worst_fitness')
        sga_best_obj = calculate_averages(sga_data, 'best_objective')
        sga_avg_obj = calculate_averages(sga_data, 'avg_objective')
        sga_worst_obj = calculate_averages(sga_data, 'worst_objective')
        
        # CHC averages
        chc_best_fit = calculate_averages(chc_data, 'best_fitness')
        chc_avg_fit = calculate_averages(chc_data, 'avg_fitness')
        chc_worst_fit = calculate_averages(chc_data, 'worst_fitness')
        chc_best_obj = calculate_averages(chc_data, 'best_objective')
        chc_avg_obj = calculate_averages(chc_data, 'avg_objective')
        chc_worst_obj = calculate_averages(chc_data, 'worst_objective')
        
        # Hill Climber statistics (best, avg, worst across runs)
        def calculate_hc_statistics(data, key, convert_to_fitness=False):
            if not data:
                return np.zeros(max_gens), np.zeros(max_gens), np.zeros(max_gens)
            
            all_series = []
            for run_data in data:
                if key in run_data:
                    series = run_data[key]
                    if convert_to_fitness:
                        # Convert objectives to fitness for hill climber
                        series = [1.0 / (1.0 + obj) for obj in series]
                    
                    # Pad series to max_gens length
                    if len(series) < max_gens:
                        # Repeat last value
                        series = series + [series[-1]] * (max_gens - len(series))
                    elif len(series) > max_gens:
                        series = series[:max_gens]
                    
                    all_series.append(series)
            
            if not all_series:
                return np.zeros(max_gens), np.zeros(max_gens), np.zeros(max_gens)
            
            # Calculate best, average, worst across runs for each generation
            best_series = np.min(all_series, axis=0)
            avg_series = np.mean(all_series, axis=0)
            worst_series = np.max(all_series, axis=0)
            
            return best_series, avg_series, worst_series
        
        # Hill Climber statistics
        hc_best_fit, hc_avg_fit, hc_worst_fit = calculate_hc_statistics(hc_data, 'objective_history', convert_to_fitness=True)
        hc_best_obj, hc_avg_obj, hc_worst_obj = calculate_hc_statistics(hc_data, 'objective_history')
        
        # Plot 1: Fitness comparison (min, avg, max for SGA and CHC)
        ax1.plot(generations[:len(sga_best_fit)], sga_best_fit[:len(generations)], 'b-', label='SGA Best', linewidth=2)
        ax1.plot(generations[:len(sga_avg_fit)], sga_avg_fit[:len(generations)], 'b--', label='SGA Average', linewidth=1)
        ax1.plot(generations[:len(sga_worst_fit)], sga_worst_fit[:len(generations)], 'b:', label='SGA Worst', linewidth=1)
        
        ax1.plot(generations[:len(chc_best_fit)], chc_best_fit[:len(generations)], 'r-', label='CHC Best', linewidth=2)
        ax1.plot(generations[:len(chc_avg_fit)], chc_avg_fit[:len(generations)], 'r--', label='CHC Average', linewidth=1)
        ax1.plot(generations[:len(chc_worst_fit)], chc_worst_fit[:len(generations)], 'r:', label='CHC Worst', linewidth=1)
        
        # Hill Climber (converted to fitness) for comparison on fitness plot
        ax1.plot(generations[:len(hc_best_fit)], hc_best_fit[:len(generations)], 'g-', label='HC Best', linewidth=2)
        ax1.plot(generations[:len(hc_avg_fit)], hc_avg_fit[:len(generations)], 'g--', label='HC Average', linewidth=1)
        ax1.plot(generations[:len(hc_worst_fit)], hc_worst_fit[:len(generations)], 'g:', label='HC Worst', linewidth=1)
        
        ax1.set_title('Population Fitness vs Generation')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Objective values comparison
        ax2.plot(generations[:len(sga_best_obj)], sga_best_obj[:len(generations)], 'b-', label='SGA Best', linewidth=2)
        ax2.plot(generations[:len(sga_avg_obj)], sga_avg_obj[:len(generations)], 'b--', label='SGA Average', linewidth=1)
        ax2.plot(generations[:len(sga_worst_obj)], sga_worst_obj[:len(generations)], 'b:', label='SGA Worst', linewidth=1)
        
        ax2.plot(generations[:len(chc_best_obj)], chc_best_obj[:len(generations)], 'r-', label='CHC Best', linewidth=2)
        ax2.plot(generations[:len(chc_avg_obj)], chc_avg_obj[:len(generations)], 'r--', label='CHC Average', linewidth=1)
        ax2.plot(generations[:len(chc_worst_obj)], chc_worst_obj[:len(generations)], 'r:', label='CHC Worst', linewidth=1)
        
        # Hill Climber objective values for comparison on objective plot
        ax2.plot(generations[:len(hc_best_obj)], hc_best_obj[:len(generations)], 'g-', label='HC Best', linewidth=2)
        ax2.plot(generations[:len(hc_avg_obj)], hc_avg_obj[:len(generations)], 'g--', label='HC Average', linewidth=1)
        ax2.plot(generations[:len(hc_worst_obj)], hc_worst_obj[:len(generations)], 'g:', label='HC Worst', linewidth=1)
        
        ax2.set_title('Population Objective Values vs Generation')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Objective Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')  # Log scale for better visualization
        
        plt.tight_layout()
        plt.savefig(f'data/{func_name}_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def print_summary_statistics(results: dict, test_functions: dict):
    """Print summary statistics for all experiments"""
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for func_name, func_info in test_functions.items():
        print(f"\n{func_info['name']}:")
        print("-" * 40)
        
        # Get final best objectives for each algorithm
        sga_finals = [run['final_best_objective'] for run in results[func_name]['sga']]
        chc_finals = [run['final_best_objective'] for run in results[func_name]['chc']]
        hc_finals = [run['final_best_objective'] for run in results[func_name]['hillclimber']]
        
        # Calculate statistics
        algorithms = [
            ('Simple GA', sga_finals),
            ('CHC GA', chc_finals),
            ('Hill Climber', hc_finals)
        ]
        
        for alg_name, finals in algorithms:
            mean_val = np.mean(finals)
            std_val = np.std(finals)
            best_val = np.min(finals)
            worst_val = np.max(finals)
            
            print(f"{alg_name:15} | Mean: {mean_val:10.6f} | Std: {std_val:10.6f} | Best: {best_val:10.6f} | Worst: {worst_val:10.6f}")
        
        # Print CHC restart statistics
        chc_restarts = [run.get('restarts', 0) for run in results[func_name]['chc']]
        print(f"CHC Restarts    | Mean: {np.mean(chc_restarts):10.2f} | Total runs with restarts: {sum(1 for r in chc_restarts if r > 0)}/30")

if __name__ == "__main__":
    print("Starting Genetic Algorithm Experiments...")
    print("This will take several minutes to complete.")
    
    # Run all experiments
    results, test_functions = run_experiments()
    
    # Create plots
    create_plots(results, test_functions)
    
    # Print summary statistics
    print_summary_statistics(results, test_functions)
    
    print(f"\nExperiments completed!")
    print(f"Results saved in 'data/' directory")
    print(f"- JSON files contain raw data for each function")
    print(f"- PNG files contain comparison plots")