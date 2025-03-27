import heapq

def search(game, start, mode, h=lambda x: 0):
    q = [(h(start), tuple(start), [])] if mode == 'a*' else [(tuple(start), [])]
    v = set()
    while q:
        item = heapq.heappop(q) if mode == 'a*' else q.pop(0 if mode == 'bfs' else -1)
        s, p = item[1], item[2] if mode == 'a*' else item
        if s in v: continue
        p += [s]
        if game.is_goal(s): return p
        v.add(s)
        for n in game.get_moves(s):
            if mode == 'a*': heapq.heappush(q, (h(n), tuple(n), p))
            else: q.append((tuple(n), p))
    return []

class TicTacToe:
    def is_goal(self, s): return s.count('X') >= 3
    def get_moves(self, s): return [s[:i] + 'X' + s[i+1:] for i in range(len(s)) if s[i] == '-']

game, start = TicTacToe(), "---"
print(f'BFS: {search(game, start, "bfs")}')
print(f'DFS: {search(game, start, "dfs")}')
print(f'A*: {search(game, start, "a*", lambda x: x.count("X"))}')
