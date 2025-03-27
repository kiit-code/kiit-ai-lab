import networkx as nx
from collections import deque

def bi_bfs(g, s, e):
    if s == e: return [s], 1
    f, b, pf, pb, ex = {s}, {e}, {s: None}, {e: None}, 0
    while f and b:
        ex += len(f)
        nf = {n for x in f for n in g[x] if n not in pf and not (pb.get(n) or pf.setdefault(n, x))}
        if nf & b:
            p = list(nf & b)
            while (x := pf.get(p[-1])): p.append(x)
            while (x := pb.get(p[0])): p.insert(0, x)
            return p[::-1], ex
        f, b, pf, pb = b, nf, pb, pf
    return [], ex

def bfs(g, s, e):
    q, v, p = deque([s]), {s}, {}
    while q:
        n = q.popleft()
        if n == e:
            r = []
            while n in p: r.append(n); n = p[n]
            return r[::-1], len(v)
        for nb in g[n]:
            if nb not in v: q.append(nb); v.add(nb); p[nb] = n
    return [], len(v)

def dfs(g, s, e):
    st, v, p = [s], set(), []
    while st:
        n = st.pop()
        if n in v: continue
        v.add(n); p.append(n)
        if n == e: return p, len(v)
        st.extend(nb for nb in g[n] if nb not in v)
    return [], len(v)

g = nx.Graph([(1,2),(2,3),(3,4),(4,5),(1,6),(6,7),(7,8),(8,5),(2,7),(3,8)])
s, e = 1, 5
print(f'Bi-BFS: {bi_bfs(g, s, e)}')
print(f'BFS: {bfs(g, s, e)}')
print(f'DFS: {dfs(g, s, e)}')
nx.draw(g, with_labels=True, node_color='lightblue', edge_color='gray')
