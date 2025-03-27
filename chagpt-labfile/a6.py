import heapq, numpy as np, matplotlib.pyplot as plt

def a_star(grid, s, g, h, m):
    pq, v = [(0, s, [])], set()
    while pq:
        c, n, p = heapq.heappop(pq)
        if n in v: continue
        p += [n]
        if n == g: return p
        v.add(n)
        for dx, dy in m:
            nxt = (n[0] + dx, n[1] + dy)
            if 0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1] and grid[nxt]:
                heapq.heappush(pq, (c + 1 + h(nxt, g), nxt, p))
    return []

def plot(grid, path):
    plt.imshow(grid, cmap='gray_r')
    plt.plot(*zip(*path), 'ro-')
    plt.show()

g = np.ones((10, 10)); g[3:7, 4] = 0
s, e = (0, 0), (9, 9)
m1, m2 = [(0,1), (1,0), (0,-1), (-1,0)], [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
h1 = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
h2 = lambda a, b: ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5
plot(g, a_star(g, s, e, h1, m1))
