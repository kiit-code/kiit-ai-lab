import heapq

import dash
import dash_bootstrap_components as dbc
import networkx as nx
import plotly.graph_objects as go

from dash import dcc, html, Input, Output, State


# ----------------------------------------------------------------------------
# 1. Create the Graph
# ----------------------------------------------------------------------------


def create_initial_graph():
    """
    Create an initial undirected graph with default weights.
    """
    G = nx.Graph()
    # Add nodes
    nodes = ["A", "B", "C", "D", "E", "F", "G"]
    for node in nodes:
        G.add_node(node)

    # Add edges (you can change these defaults)
    edges = [
        ("A", "B", 1),
        ("A", "C", 4),
        ("B", "D", 2),
        ("B", "E", 5),
        ("C", "F", 2),
        ("D", "G", 3),
        ("E", "G", 1),
        ("F", "G", 6),
    ]
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)  # undirected edge

    return G


graph = create_initial_graph()

# Set default heuristics for each node
default_heuristics = {
    "A": 6,
    "B": 4,
    "C": 5,
    "D": 3,
    "E": 2,
    "F": 3,
    "G": 0,  # Usually 0 for the goal node in A*
}


# ----------------------------------------------------------------------------
# 2. Implement the Search Algorithms
# ----------------------------------------------------------------------------


def astar_search(graph, start, goal, heuristics):
    """
    Perform the A* search from start to goal on the given graph
    with the specified heuristics dict.
    """
    # Priority queue: (f_score, node, path, g_score)
    open_set = [(heuristics[start], start, [start], 0)]
    heapq.heapify(open_set)

    # Keep track of best g_scores
    g_scores = {start: 0}

    while open_set:
        f_score, current, path, g_score = heapq.heappop(open_set)

        if current == goal:
            return path, g_score

        for neighbor in graph.neighbors(current):
            tentative_g = g_score + graph[current][neighbor]["weight"]

            if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g
                f_score = tentative_g + heuristics[neighbor]
                new_path = path + [neighbor]
                heapq.heappush(open_set, (f_score, neighbor, new_path, tentative_g))

    return None, float("inf")


def ucs_search(graph, start, goal):
    """
    Perform the Uniform Cost Search (Dijkstra-like) from start to goal.
    """
    # Priority queue: (cost_so_far, node, path)
    open_list = []
    heapq.heappush(open_list, (0, start, [start]))
    visited = set()

    while open_list:
        cost_so_far, current_node, path = heapq.heappop(open_list)

        if current_node == goal:
            return path, cost_so_far

        if current_node in visited:
            continue
        visited.add(current_node)

        for neighbor in graph.neighbors(current_node):
            edge_weight = graph[current_node][neighbor]["weight"]
            new_cost = cost_so_far + edge_weight
            new_path = path + [neighbor]
            heapq.heappush(open_list, (new_cost, neighbor, new_path))

    return None, float("inf")


# ----------------------------------------------------------------------------
# 3. Build the Dash App
# ----------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# Prepare lists for dynamic form creation
edge_list = list(graph.edges(data=True))  # e.g. [('A','B',{'weight':1}), ...]
node_list = list(graph.nodes())  # e.g. ['A','B','C','D','E','F','G']

# Define the layout
app.layout = html.Div(
    [
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H1(
                                    "Pathfinding Visualization",
                                    className="text-center mb-4",
                                ),
                                html.P(
                                    "Compare A* and Uniform Cost Search (UCS) algorithms on an interactive graph.",
                                    className="text-center text-muted mb-4",
                                ),
                            ]
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    "Select Nodes",
                                                    className="card-title",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Label(
                                                                    "Start Node",
                                                                    className="font-weight-bold",
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="start-node",
                                                                    options=[
                                                                        {
                                                                            "label": node,
                                                                            "value": node,
                                                                        }
                                                                        for node in node_list
                                                                    ],
                                                                    value="A",
                                                                    clearable=False,
                                                                ),
                                                            ],
                                                            md=6,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.Label(
                                                                    "Goal Node",
                                                                    className="font-weight-bold",
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="goal-node",
                                                                    options=[
                                                                        {
                                                                            "label": node,
                                                                            "value": node,
                                                                        }
                                                                        for node in node_list
                                                                    ],
                                                                    value="G",
                                                                    clearable=False,
                                                                ),
                                                            ],
                                                            md=6,
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        )
                                    ],
                                    className="mb-4",
                                ),
                            ],
                            md=12,
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    "Edge Weights",
                                                    className="card-title",
                                                ),
                                                html.Div(
                                                    [
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        html.Label(
                                                                            f"Edge {u} → {v}",
                                                                            className="font-weight-bold",
                                                                        ),
                                                                        dbc.Input(
                                                                            id=f"weight-{u}-{v}",
                                                                            type="number",
                                                                            value=data[
                                                                                "weight"
                                                                            ],
                                                                            min=1,
                                                                            step=1,
                                                                        ),
                                                                    ],
                                                                    md=4,
                                                                )
                                                                for (
                                                                    u,
                                                                    v,
                                                                    data,
                                                                ) in edge_list
                                                            ],
                                                            className="g-2",
                                                        )
                                                    ]
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            md=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    "Heuristic Values",
                                                    className="card-title",
                                                ),
                                                html.Div(
                                                    [
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        html.Label(
                                                                            f"Node {node}",
                                                                            className="font-weight-bold",
                                                                        ),
                                                                        dbc.Input(
                                                                            id=f"heuristic-{node}",
                                                                            type="number",
                                                                            value=default_heuristics[
                                                                                node
                                                                            ],
                                                                            min=0,
                                                                            step=1,
                                                                        ),
                                                                    ],
                                                                    md=4,
                                                                )
                                                                for node in node_list
                                                            ],
                                                            className="g-2",
                                                        )
                                                    ]
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            md=6,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Button(
                                    "Run Pathfinding",
                                    id="run-button",
                                    color="primary",
                                    size="lg",
                                    className="w-100 mb-4",
                                ),
                            ],
                            md=12,
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4("Results"),
                                                html.Div(
                                                    id="results-astar",
                                                    className="alert alert-info",
                                                ),
                                                html.Div(
                                                    id="results-ucs",
                                                    className="alert alert-success",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="mb-4",
                                )
                            ],
                            md=12,
                        )
                    ]
                ),
                dbc.Row([dbc.Col([dcc.Graph(id="graph-visualization")], md=12)]),
            ],
            fluid=True,
        )
    ]
)


# ----------------------------------------------------------------------------
# 4. Define Callbacks
# ----------------------------------------------------------------------------


@app.callback(
    Output("results-astar", "children"),
    Output("results-ucs", "children"),
    Output("graph-visualization", "figure"),
    [Input("run-button", "n_clicks")],
    [State("start-node", "value"), State("goal-node", "value")]
    + [State(f"weight-{u}-{v}", "value") for (u, v, d) in edge_list]
    + [State(f"heuristic-{node}", "value") for node in node_list],
)


def update_dashboard(n_clicks, start, goal, *args):
    """
    1) The first len(edge_list) values correspond to new edge weights.
    2) The next len(node_list) values correspond to new heuristics.
    """
    if not n_clicks:
        return "", "", go.Figure()

    # Separate out edge weight values from heuristics
    weights = args[: len(edge_list)]
    heuristics_values = args[len(edge_list) :]

    # 1. Update edge weights in the graph
    for i, (u, v, d) in enumerate(edge_list):
        graph[u][v]["weight"] = weights[i]
        # For undirected graph, also set the reverse edge
        if not isinstance(graph, nx.DiGraph):
            graph[v][u]["weight"] = weights[i]

    # 2. Update heuristics
    heuristics = {}
    for i, node in enumerate(node_list):
        heuristics[node] = heuristics_values[i]

    # 3. Run A* and UCS
    path_astar, cost_astar = astar_search(graph, start, goal, heuristics)
    path_ucs, cost_ucs = ucs_search(graph, start, goal)

    # Build result strings
    if path_astar:
        results_astar = f"A* Path: {path_astar}, Total Cost: {cost_astar}"
    else:
        results_astar = "A* could not find a path."

    if path_ucs:
        results_ucs = f"UCS Path: {path_ucs}, Total Cost: {cost_ucs}"
    else:
        results_ucs = "UCS could not find a path."

    # 4. Create a figure
    fig = draw_graph_with_path(graph, path_astar, path_ucs)

    return results_astar, results_ucs, fig


# ----------------------------------------------------------------------------
# 5. Helper Function for Plotly Graph
# ----------------------------------------------------------------------------


def draw_graph_with_path(graph, path_astar, path_ucs):
    """
    Create a plotly figure of the graph, highlighting edges used by A* and UCS.
    """
    pos = nx.spring_layout(graph, seed=42)

    # Create edge traces
    edge_traces = []

    # Regular edges
    for u, v, data in graph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=1, color="#888"),
                hoverinfo="text",
                text=f'Weight: {data["weight"]}',
                mode="lines",
                showlegend=False,
            )
        )

    # A* path
    if path_astar:
        astar_x = []
        astar_y = []
        for i in range(len(path_astar) - 1):
            u, v = path_astar[i], path_astar[i + 1]
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            astar_x.extend([x0, x1, None])
            astar_y.extend([y0, y1, None])
        edge_traces.append(
            go.Scatter(
                x=astar_x,
                y=astar_y,
                line=dict(width=3, color="rgba(255,0,0,0.7)"),
                hoverinfo="text",
                text="A* path",
                mode="lines",
                name="A* Path",
            )
        )

    # UCS path
    if path_ucs:
        ucs_x = []
        ucs_y = []
        for i in range(len(path_ucs) - 1):
            u, v = path_ucs[i], path_ucs[i + 1]
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ucs_x.extend([x0, x1, None])
            ucs_y.extend([y0, y1, None])
        edge_traces.append(
            go.Scatter(
                x=ucs_x,
                y=ucs_y,
                line=dict(width=3, color="rgba(0,255,0,0.7)"),
                hoverinfo="text",
                text="UCS path",
                mode="lines",
                name="UCS Path",
            )
        )

    # Create node trace
    node_trace = go.Scatter(
        x=[pos[node][0] for node in graph.nodes()],
        y=[pos[node][1] for node in graph.nodes()],
        mode="markers+text",
        text=list(graph.nodes()),
        textposition="top center",
        hoverinfo="text",
        marker=dict(size=30, color="lightblue", line=dict(width=2, color="darkblue")),
        name="Nodes",
    )

    # Create the figure
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
        ),
    )

    return fig


# ----------------------------------------------------------------------------
# 6. Run the App
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    app.run_server(debug=True)
