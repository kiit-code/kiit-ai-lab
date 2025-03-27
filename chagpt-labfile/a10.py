import time, heapq, networkx as nx, matplotlib.pyplot as plt

graph = {
    'S': {'A': 1, 'B': 4}, 'A': {'C': 2, 'D': 5}, 'B': {'D': 1},
    'C': {'G': 3}, 'D': {'G': 2}, 'G': {}
}

def search(algo):
    frontier = [(0, ['S'])] if algo in ['ucs', 'best', 'a*'] else [['S']]
    explored, steps, came_from = set(), 0, {}
    while frontier:
        path = heapq.heappop(frontier)[1] if algo in ['ucs', 'best', 'a*'] else frontier.pop(0 if algo in ['bfs', 'bi-bfs'] else -1)
        if isinstance(path, str): path = [path]  # Ensure path is a list
        node = path[-1]
        if node == 'G': return steps, len(explored), reconstruct_path(came_from, 'G')
        if node in explored: continue
        explored.add(node)
        steps += 1
        for neighbor, cost in graph[node].items():
            if neighbor not in explored:
                new_path = list(path) + [neighbor]  # Ensure new_path is a list
                heapq.heappush(frontier, (cost, new_path)) if algo in ['ucs', 'best'] else frontier.append(new_path)
                came_from[neighbor] = node
    return float('inf'), len(explored), []

def reconstruct_path(came_from, end):
    path = [end]
    while path[-1] in came_from:
        path.append(came_from[path[-1]])
    return path[::-1]

def compare():
    algos = ['bfs', 'dfs', 'bi-bfs', 'ucs', 'best', 'a*']
    results = {algo: search(algo) for algo in algos}
    plt.bar(results.keys(), [res[1] for res in results.values()])
    plt.title('Nodes Explored Comparison')
    plt.show()
    return results

print(compare())
