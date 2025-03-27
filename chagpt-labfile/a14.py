import numpy as np

def valid(b, r, c): return all(b[i] != c and abs(b[i] - c) != r - i for i in range(r))

def moves(b, t): return [(b[:t] + [c] + b[t+1:], t+1) for c in range(4) if valid(b, t, c)]

def minimax(b, t, a, bta, mx):
    if t == 4: return 100 if mx else -100
    best = -1e9 if mx else 1e9
    for mv, nt in moves(b, t):
        s = minimax(mv, nt, a, bta, not mx)
        best = max(best, s) if mx else min(best, s)
        if mx: a = max(a, best)
        else: bta = min(bta, best)
        if bta <= a: break
    return best

def best_move(b, t): return max(moves(b, t), key=lambda mv: minimax(mv[0], mv[1], -1e9, 1e9, False))[0]

def display(b):
    g = np.full((4, 4), '.')
    for r, c in enumerate(b):
        if c != -1: g[r][c] = 'Q'
    print('\n'.join(' '.join(row) for row in g) + '\n')

def play():
    b, t = [-1] * 4, 0
    while t < 4:
        display(b)
        b[t] = int(input("Col(0-3): ")) if t % 2 == 0 else best_move(b, t)[t]
        t += 1
    display(b)
    print("Game Over!")

play()
