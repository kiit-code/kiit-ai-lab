import numpy as np
from collections import deque

def bfs(maze, start, end):
    q, visited, parent = deque([start]), {start}, {}
    while q:
        x, y = q.popleft()
        if (x, y) == end:
            path = []
            while (x, y) in parent:
                path.append((x, y))
                x, y = parent[(x, y)]
            return path[::-1], len(visited)
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            nx, ny = x+dx, y+dy
            if (0 <= nx < 5 and 0 <= ny < 5 and maze[nx][ny] and (nx, ny) not in visited):
                q.append((nx, ny))
                visited.add((nx, ny))
                parent[(nx, ny)] = (x, y)
    return [], len(visited)

def dfs(maze, start, end):
    stack, visited, path = [start], set(), []
    while stack:
        x, y = stack.pop()
        if (x, y) in visited:
            continue
        visited.add((x, y))
        path.append((x, y))
        if (x, y) == end:
            return path, len(visited)
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < 5 and 0 <= ny < 5 and maze[nx][ny] and (nx, ny) not in visited:
                stack.append((nx, ny))
    return [], len(visited)

maze = np.array([[1,1,0,0,1],
                 [0,1,0,1,1],
                 [0,1,1,1,0],
                 [1,0,0,1,1],
                 [1,1,1,0,1]])

start, end = (0,0), (4,4)
bfs_path, bfs_explored = bfs(maze, start, end)
dfs_path, dfs_explored = dfs(maze, start, end)

print(f'BFS: {bfs_path}, Explored: {bfs_explored}')
print(f'DFS: {dfs_path}, Explored: {dfs_explored}')
