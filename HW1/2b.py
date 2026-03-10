def U_pi1(alpha: float, R: float) -> float:
    # pi1: -1, -1, -1, (R-1)
    return R - 4

def U_pi2(alpha: float, R: float) -> float:
    # pi2: -1, -1, then brake:
    # success: (R-1) with prob (1-alpha)
    # fail:    (-1)  with prob alpha
    return (1 - alpha) * R - 3

def better_region(alpha: float) -> str:
    return f"pi2 better than pi1 when alpha*R < 1  (i.e., R < {1/alpha:.4f})"

if __name__ == "__main__":
    alpha = 0.1
    R = 15
    print("U(pi1) =", U_pi1(alpha, R))
    print("U(pi2) =", U_pi2(alpha, R))
    print(better_region(alpha))
