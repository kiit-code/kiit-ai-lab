from heapq import heappop, heappush

tasks = {"A": (3, []), "B": (2, ["A"]), "C": (1, ["A"]), "D": (4, ["B", "C"])}


def heuristic(remaining):
    return sum(tasks[t][0] for t in remaining)


def a_star():
    queue, visited = [(0, [], set(tasks.keys()))], {}
    while queue:
        cost, path, remaining = heappop(queue)
        if not remaining:
            return cost, path
        for t in remaining:
            if all(dep in path for dep in tasks[t][1]):
                new_path = path + [t]
                new_remaining = remaining - {t}
                g = cost + tasks[t][0]
                f = g + heuristic(new_remaining)
                if (
                    tuple(new_remaining) not in visited
                    or visited[tuple(new_remaining)] > f
                ):
                    visited[tuple(new_remaining)] = f
                    heappush(queue, (f, new_path, new_remaining))


def greedy():
    order, done = [], set()
    while len(order) < len(tasks):
        t = min(
            (t for t in tasks if t not in done and all(d in done for d in tasks[t][1])),
            key=lambda x: tasks[x][0],
        )
        order.append(t)
        done.add(t)
    return sum(tasks[t][0] for t in order), order


print(a_star())
print(greedy())
