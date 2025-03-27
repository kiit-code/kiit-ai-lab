import heapq
import numpy as np

def h1(state, goal):
    return sum(state[i] != goal[i] for i in range(9) if state[i] != 0)

def h2(state, goal):
    return sum(abs(i//3 - goal.index(state[i])//3) + abs(i%3 - goal.index(state[i])%3) for i in range(9) if state[i] != 0)

def a_star(start, goal, heuristic):
    pq, visited = [(heuristic(start, goal), 0, start, [])], set()
    while pq:
        _, cost, state, path = heapq.heappop(pq)
        if state == goal: return path + [state], len(visited)
        visited.add(tuple(state))
        zero = state.index(0)
        for move in [-1, 1, -3, 3]:
            new_zero = zero + move
            if new_zero in range(9) and (zero%3-new_zero%3)**2 != 4:
                new_state = state[:]
                new_state[zero], new_state[new_zero] = new_state[new_zero], new_state[zero]
                if tuple(new_state) not in visited:
                    heapq.heappush(pq, (cost + 1 + heuristic(new_state, goal), cost + 1, new_state, path + [state]))
    return [], len(visited)

start = [1, 2, 3, 4, 0, 5, 6, 7, 8]
goal = [1, 2, 3, 4, 5, 6, 7, 8, 0]
print(f'H1: {a_star(start, goal, h1)}')
print(f'H2: {a_star(start, goal, h2)}')
