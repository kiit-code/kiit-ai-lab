import copy

def minimax(s, d, m, a, b):
    if s[-1] == [3, 2, 1]: return 100 if m else -100
    if d == 0: return sum(len(r) for r in s)
    best = -1e9 if m else 1e9
    for mv in moves(s):
        v = minimax(mv, d-1, not m, a, b)
        best = max(best, v) if m else min(best, v)
        if m: a = max(a, best)
        else: b = min(b, best)
        if b <= a: break
    return best

def moves(s):
    return [copy.deepcopy(s) for i, r in enumerate(s) if r for j, d in enumerate(s) if i != j and (not d or r[-1] < d[-1]) and (s := copy.deepcopy(s)) and (s[j].append(s[i].pop()))]

def play():
    s, p = [[3, 2, 1], [], []], 1
    while True:
        [print(r) for r in s]
        if s[-1] == [3, 2, 1]: return print("AI Wins!" if not p else "You Win!")
        if p:
            mv = list(map(int, input("Move: ").split()))
            if mv in [(i, j) for i in range(3) for j in range(3) if i != j]: s[mv[1]].append(s[mv[0]].pop())
            else: print("Invalid."); continue
        else:
            s = max(moves(s), key=lambda m: minimax(m, 3, 0, -1e9, 1e9))
        p ^= 1

play()
