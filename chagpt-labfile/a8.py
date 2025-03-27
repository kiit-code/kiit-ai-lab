import heapq

def search(grid, s, g, c=None):
    pq, v = [(0, s, [])], set()
    while pq:
        cost, p, path = heapq.heappop(pq) if c else pq.pop(0)
        if p in v: continue
        path.append(p)
        if p in g: g.remove(p)
        if not g: return path, cost
        v.add(p)
        for d in [(0,1), (1,0), (0,-1), (-1,0)]:
            nxt = (p[0]+d[0], p[1]+d[1])
            if 0 <= nxt[0] < len(grid) and 0 <= nxt[1] < len(grid[0]) and grid[nxt[0]][nxt[1]]:
                heapq.heappush(pq, (cost + (c.get(nxt, 1) if c else 0), nxt, path[:])) if c else pq.append((nxt, path[:]))
    return [], float('inf')

g = [[1]*5 for _ in range(5)]
s, goals, costs = (0,0), {(4,4), (2,2)}, {(4,4): 2, (2,2): 1}
print('BFS:', search(g, s, goals.copy()))
print('UCS:', search(g, s, goals.copy(), costs))