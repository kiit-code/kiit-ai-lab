# /// script
# dependencies = ["pandas"]
# ///

import heapq
import pandas as pd


# Define the graph structure with costs
graph = {
    'S': [('A', 3), ('D', 4)],
    'A': [('B', 4)],
    'B': [('C', 4), ('E', 5)],
    'C': [('G', 3.4)],
    'D': [('E', 2)],
    'E': [('F', 4)],
    'F': [('G', 3.5)]
}

# Define heuristic values (h(n))
heuristic = {
    'S': 11.5, 'A': 10.1, 'B': 5.8, 'C': 3.4, 'D': 9.2,
    'E': 7.1, 'F': 3.5, 'G': 0
}

# Implementing A* Algorithm
def a_star_search(graph, start, goal, heuristic):
    open_list = []
    heapq.heappush(open_list, (0 + heuristic[start], 0, start, [start]))  # (f, g, node, path)
    visited = {}

    while open_list:
        f, g, current, path = heapq.heappop(open_list)

        if current in visited and visited[current] <= g:
            continue

        visited[current] = g

        if current == goal:
            return path, g

        for neighbor, cost in graph.get(current, []):
            new_g = g + cost
            new_f = new_g + heuristic[neighbor]
            heapq.heappush(open_list, (new_f, new_g, neighbor, path + [neighbor]))

    return None, float('inf')

# Find the optimal path using A*
optimal_path, total_cost = a_star_search(graph, 'S', 'G', heuristic)

# Prepare DataFrame for visualization
df = pd.DataFrame(columns=["Step", "Node", "g(n)", "h(n)", "f(n)", "Path"])
g_cost = 0
path_trace = []

for step, node in enumerate(optimal_path):
    path_trace.append(node)
    h_cost = heuristic[node]
    f_cost = g_cost + h_cost
    df = pd.concat([df, pd.DataFrame([{
    "Step": step + 1,
    "Node": node,
    "g(n)": g_cost,
    "h(n)": h_cost,
    "f(n)": f_cost,
    "Path": " → ".join(path_trace)
    }])], ignore_index=True)
    
    if step < len(optimal_path) - 1:
        g_cost += dict(graph[node])[optimal_path[step + 1]]

# Display the DataFrame
print(df)