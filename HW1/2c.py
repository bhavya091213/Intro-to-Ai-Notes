def V_pi1_n11(R: float) -> float:
    # 10 total steps to reach (11,1): 9 steps give -1, last gives (R-1)
    return (R - 1) - 9

def V_pi2_n11(alpha: float, R: float) -> float:
    # Branching described in write-up:
    # 0.9 -> total 9 steps, success return = 14-8 = 6
    # 0.1 -> reach (10,2), then:
    #        0.9 -> total 8 steps success return = 14-7 = 7
    #        0.1 -> total 8 steps fail return = -8
    success_9_steps = (R - 1) - 8
    success_8_steps = (R - 1) - 7
    fail_8_steps = -8

    return (1 - alpha) * success_9_steps + alpha * (
        (1 - alpha) * success_8_steps + alpha * fail_8_steps
    )

if __name__ == "__main__":
    alpha = 0.1
    R = 15
    print("V_pi1(start) =", V_pi1_n11(R))
    print("V_pi2(start) =", V_pi2_n11(alpha, R))
