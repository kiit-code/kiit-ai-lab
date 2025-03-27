import random


def minimax(s, d, m, a, b, t, c):
    if t in s:
        return 100 if m else -100
    if d == 0:
        return abs(t - max(s)) - abs(t - min(s))
    v = -float("inf") if m else float("inf")
    for mv in moves(s, c):
        v = (
            max(v, minimax(mv, d - 1, 0, a, b, t, c))
            if m
            else min(v, minimax(mv, d - 1, 1, a, b, t, c))
        )
        if m:
            a = max(a, v)
        else:
            b = min(b, v)
        if b <= a:
            break
    return v


def moves(s, c):
    x, y, p, q = *s, min(s[0], c[1] - s[1]), min(s[1], c[0] - s[0])
    return {(c[0], y), (x, c[1]), (0, y), (x, 0), (x - p, y + p), (x + q, y - q)}


def best(s, c, t):
    return max(moves(s, c), key=lambda m: minimax(m, 5, 0, -1e9, 1e9, t, c))


def play():
    s, c, t, p = (0, 0), (4, 3), 2, 1
    while 1:
        print(f"State: {s}")
        if t in s:
            return print("AI Wins!" if not p else "You Win!")
        s = tuple(map(int, input("Move: ").split())) if p else best(s, c, t)
        p ^= 1


play()
