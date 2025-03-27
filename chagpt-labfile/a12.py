import random

def fitness(x): return -(x - 42) ** 2

def mutate(x): return x + random.choice([-1, 1])

def crossover(a, b): return (a + b) // 2

def genetic_algorithm():
    population = [random.randint(0, 100) for _ in range(10)]
    for _ in range(100):
        population.sort(key=fitness, reverse=True)
        if fitness(population[0]) == 0: return population[0]
        new_gen = [population[0], population[1]] + [mutate(crossover(population[i], population[i+1])) for i in range(4)]
        population = new_gen + [random.randint(0, 100) for _ in range(4)]
    return max(population, key=fitness)

print(genetic_algorithm())