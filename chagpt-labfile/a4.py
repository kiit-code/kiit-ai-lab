import heapq

def ucs(graph, start, end):
    pq, visited = [(0, start, [])], set()
    while pq:
        cost, node, path = heapq.heappop(pq)
        if node in visited: continue
        path = path + [node]
        if node == end: return path, cost
        visited.add(node)
        for neighbor, weight in graph.get(node, []):
            if neighbor not in visited:
                heapq.heappush(pq, (cost + weight, neighbor, path))
    return [], float('inf')

def bfs(graph, start, end):
    q, visited = [(start, [start])], set()
    while q:
        node, path = q.pop(0)
        if node == end: return path
        visited.add(node)
        for neighbor, _ in graph.get(node, []):
            if neighbor not in visited:
                q.append((neighbor, path + [neighbor]))
    return []

graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('C', 2), ('D', 5)],
    'C': [('D', 1)],
    'D': []
}

start, end = 'A', 'D'
print(f'UCS: {ucs(graph, start, end)}')
print(f'BFS: {bfs(graph, start, end)}')
