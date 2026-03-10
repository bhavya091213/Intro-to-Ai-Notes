from __future__ import annotations

ETA = 0.5
GAMMA = 1.0

ACTIONS = ["Walk", "Bus"]

transitions = [
    (1, "Bus", -1, 1),
    (1, "Bus", -1, 1),
    (1, "Bus", 3, 3),
    (3, "Walk", 1, 4),
    (4, "Walk", 1, 5),
]

Q = {}  # (s,a) -> value


def get_q(s, a):
    return Q.get((s, a), 0.0)


def max_q(s):
    return max(get_q(s, a) for a in ACTIONS)


def update(s, a, r, sp):
    old = get_q(s, a)
    target = r + GAMMA * max_q(sp)
    new = old + ETA * (target - old)
    Q[(s, a)] = new
    return new


def main():
    q1_after_step1 = None

    for i, (s, a, r, sp) in enumerate(transitions, start=1):
        new_val = update(s, a, r, sp)
        print(f"Step {i}: ({s}, {a}, {r}, {sp}) -> Q({s},{a}) = {new_val}")
        if i == 1:
            q1_after_step1 = new_val

    print("\nFinal requested values:")
    print("Q^(1, Bus) after step 1:", q1_after_step1)     # -0.5
    print("Q^(1, Bus) after full pass:", get_q(1, "Bus")) # 1.125
    print("Q^(3, Walk):", get_q(3, "Walk"))               # 0.5
    print("Q^(4, Walk):", get_q(4, "Walk"))               # 0.5


if __name__ == "__main__":
    main()
