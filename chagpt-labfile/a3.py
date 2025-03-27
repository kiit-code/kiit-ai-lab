import heapq
import numpy as np

def best_first_search(grid, start, treasure):
    h = lambda x, y: abs(x - treasure[0]) + abs(y - treasure[1])
    pq, visited = [(h(*start), start)], set()
    while pq:
        _, (x, y) = heapq.heappop(pq)
        if (x, y) == treasure: return (x, y)
        visited.add((x, y))
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < 5 and 0 <= ny < 5 and (nx, ny) not in visited:
                heapq.heappush(pq, (h(nx, ny), (nx, ny)))
    return None

grid = np.random.randint(1, 10, (5,5))
start, treasure = (0, 0), (4, 4)
print(f'Treasure found at: {best_first_search(grid, start, treasure)}')
