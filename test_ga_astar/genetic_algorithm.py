import numpy as np
import random

# --------------------------------------------------
# 1. Problem Constants & GA Hyperparameters
# --------------------------------------------------

# Ranges for parameters
F_MAX = 1000.0          # N (example)
THETA_MIN, THETA_MAX = -180.0, 180.0
T_MAX = 60.0            # s
TIGN_MAX = 120.0        # s

# Weights in the cost function
w1 = 1.0  # weight for Docking Error
w2 = 1.0  # weight for Fuel Consumption
w3 = 1.0  # weight for Safety Penalty

# Constraints
EPSILON = 0.1          # max allowed docking error in meters
V_SAFE = 0.5           # m/s
FUEL_MAX = 100.0       # arbitrary
D_SAFE = 10.0          # m

# GA parameters
POP_SIZE = 30
N_GENERATIONS = 50
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 3

# --------------------------------------------------
# 2. Representation
# --------------------------------------------------

# Each individual is represented as a dict or array with 4 entries: [F, theta, t, t_ign].


# --------------------------------------------------
# 3. Initialization
# --------------------------------------------------

def create_individual():
    """
    Randomly create a valid individual (chromosome).
    """
    individual = {
        'F': random.uniform(0.0, F_MAX),
        'theta': random.uniform(THETA_MIN, THETA_MAX),
        't': random.uniform(0.0, T_MAX),
        'tign': random.uniform(0.0, TIGN_MAX)
    }
    return individual

def initialize_population(pop_size):
    """
    Create the initial population with pop_size individuals.
    """
    return [create_individual() for _ in range(pop_size)]


# --------------------------------------------------
# 4. Cost (Objective) Function
# --------------------------------------------------

def simulate_docking(individual):
    """
    Placeholder function to simulate docking with the given parameters.
    Returns:
        docking_error, fuel_consumption, safety_penalty
    In a real scenario, you would run a physics-based simulation here.
    """
    F = individual['F']
    theta = individual['theta']
    burn_time = individual['t']
    ignition_time = individual['tign']
    
    # Placeholder logic for demonstration:
    # We'll pretend that docking error is somehow proportional to the absolute difference
    # from some "ideal" thrust angle, random factors, etc. 
    # You MUST replace this with your real simulation code.

    # Let's create some dummy values:
    # * docking_error in [0, 2]
    # * fuel_consumption in [0, FUEL_MAX] (roughly)
    # * safety_penalty in [0, 10]
    
    # relationships:
    docking_error = abs(theta) / 180.0  # just a dummy measure
    fuel_consumption = (F / F_MAX) * burn_time  # simplistic approach
    
    # Safety penalty if it violates constraints 
    # (distance, velocity, or other constraints).
    # We'll artificially model a penalty if we use too high F or burn too long, etc.
    safety_penalty = 0.0
    
    # Check constraints
    # 1. If final docking error > EPSILON, add penalty
    if docking_error > EPSILON:
        safety_penalty += (docking_error - EPSILON) * 10.0
    
    
    # 2. If fuel consumption > FUEL_MAX, large penalty
    if fuel_consumption > FUEL_MAX:
        safety_penalty += 1000.0  # big penalty
    
    
    # 3. If any "imaginary" safety distance is violated:
    if F > 0.8 * F_MAX and burn_time > 0.8 * T_MAX:
        safety_penalty += 50.0
    
    return docking_error, fuel_consumption, safety_penalty

def evaluate_individual(individual):
    """
    Calculate the cost for an individual:
        Cost = w1 * Docking Error + w2 * Fuel Consumption + w3 * Safety Penalty
    """
    docking_error, fuel_consumption, safety_penalty = simulate_docking(individual)
    cost = w1 * docking_error + w2 * fuel_consumption + w3 * safety_penalty
    return cost



# Selection, Crossover, Mutation
def tournament_selection(pop, k=TOURNAMENT_SIZE):
    """
    Tournament selection: randomly choose k individuals and return the best (lowest cost).
    """
    selected = random.sample(pop, k)
    best = min(selected, key=lambda ind: ind['cost'])
    return best

def crossover(parent1, parent2):
    """
    Single-point or arithmetic crossover on parameters.
    We'll do a simple arithmetic blend for demonstration.
    """
    child1 = {}
    child2 = {}
    
    alpha = random.random()
    
    # simple linear combination for each parameter
    for param in ['F', 'theta', 't', 'tign']:
        child1[param] = alpha * parent1[param] + (1 - alpha) * parent2[param]
        child2[param] = alpha * parent2[param] + (1 - alpha) * parent1[param]
    
    return child1, child2

def mutate(individual, mutation_rate=MUTATION_RATE):
    """
    Randomly perturb each parameter with some probability.
    """
    if random.random() < mutation_rate:
        # Mutate thrust magnitude
        individual['F'] = random.uniform(0.0, F_MAX)
    if random.random() < mutation_rate:
        # Mutate angle
        individual['theta'] = random.uniform(THETA_MIN, THETA_MAX)
    if random.random() < mutation_rate:
        # Mutate burn duration
        individual['t'] = random.uniform(0.0, T_MAX)
    if random.random() < mutation_rate:
        # Mutate ignition time
        individual['tign'] = random.uniform(0.0, TIGN_MAX)
    return individual


# --------------------------------------------------
# 6. Main GA Loop
# --------------------------------------------------

def genetic_algorithm():
    

    population = initialize_population(POP_SIZE)
    
    for ind in population:
        ind['cost'] = evaluate_individual(ind)
    
    best_solution = min(population, key=lambda ind: ind['cost'])
    
    for gen in range(N_GENERATIONS):
        new_population = []
        
        while len(new_population) < POP_SIZE:
            # Selection
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutation
            child1 = mutate(child1)
            child2 = mutate(child2)
            
            # Evaluate children
            child1['cost'] = evaluate_individual(child1)
            child2['cost'] = evaluate_individual(child2)
            
            new_population.extend([child1, child2])
        

        population = new_population
        
        # Check for new best solution
        current_best = min(population, key=lambda ind: ind['cost'])
        if current_best['cost'] < best_solution['cost']:
            best_solution = current_best
        
        # (Optional) Print out info each generation
        print(f"Generation {gen+1}/{N_GENERATIONS}, Best Cost: {best_solution['cost']:.4f}")
    
    return best_solution


# --------------------------------------------------
# 7. Run the GA
# --------------------------------------------------

if __name__ == "__main__":
    solution = genetic_algorithm()
    print("\nBest solution found:")
    print(f" - F     = {solution['F']:.2f} N")
    print(f" - theta = {solution['theta']:.2f} deg")
    print(f" - t     = {solution['t']:.2f} s")
    print(f" - tign  = {solution['tign']:.2f} s")
    print(f"Cost = {solution['cost']:.4f}")