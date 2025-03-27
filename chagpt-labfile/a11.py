import random

def hill_climb(f, neighbors, start):
    current = start
    while True:
        next_move = max(neighbors(current), key=f, default=None)
        if next_move is None or f(next_move) <= f(current):
            return current
        current = next_move

def fitness(x): return -(x - 3) ** 2 + 9

def neighbors(x): return [x + 1, x - 1] if 0 <= x <= 6 else []

print(hill_climb(fitness, neighbors, random.randint(0, 6)))
