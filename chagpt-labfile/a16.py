import heapq

goal = [1, 2, 3, 4, 5, 6, 7, 8, 0]

def heuristic(state, method):
    return sum(s != g for s, g in zip(state, goal)) if method == "misplaced" else sum(abs(b % 3 - g % 3) + abs(b // 3 - g // 3) for b, g in ((state.index(i), goal.index(i)) for i in range(1, 9)))

def moves(state):
    i = state.index(0)
    for d in (-3, 3, -1, 1):
        if 0 <= i + d < 9 and not (i % 3 == 2 and d == 1) and not (i % 3 == 0 and d == -1):
            new_state = state[:]
            new_state[i], new_state[i + d] = new_state[i + d], new_state[i]
            yield new_state

def search(start, method, use_g):
    heap, seen = [(0, start, 0, [])], {tuple(start)}
    while heap:
        f, state, g, path = heapq.heappop(heap)
        if state == goal: return path + [state]
        for move in moves(state):
            if tuple(move) not in seen:
                seen.add(tuple(move))
                h = heuristic(move, method)
                heapq.heappush(heap, ((g + 1 + h) if use_g else h, move, g + 1, path + [state]))
    return None

def solve(start):
    return {algo: search(start, method, algo == "A*") for algo in ["A*", "GBFS"] for method in ["misplaced", "manhattan"]}

start = [1, 2, 3, 4, 0, 5, 7, 8, 6]
print(solve(start))
