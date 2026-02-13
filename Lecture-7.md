# Lecture 7

## Recap
- Temporal Distance Learning
  - updating in each cycle what you learn over time, defining learning rates to see how new information effects previous values


## Active Reinforcment Learning
- You choose actions to take, optimal policies is the goat
- We don't know transitions and rewards, goal is to learn optimal policy and values
- Learner is making the choices
- Fundamental tradeoff of exploration vs explotation of actions, exploit current good actions or explore new actions
  - gemini explain this more in depth
    - Gemini response:
      Active Reinforcement Learning involves a learner that actively chooses actions to take in order to learn an optimal policy and values, especially when transitions and rewards are unknown. A fundamental challenge in this process is the tradeoff between exploration (trying new actions to discover potentially better outcomes) and exploitation (sticking with actions that are currently known to be good). This tradeoff is crucial because exploring too much might lead to suboptimal performance in the short term, while exploiting too much could mean missing out on discovering a truly optimal strategy.

## From Value iteration to Q value iteration
- Value iteration finds sucessive values (this is from MDP)
- Gemini give the recurrence relation here and explain it
  - Gemini response:
    The recurrence relation for Q-learning is:
    Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a'(Q(s', a')) - Q(s, a)]

    Explanation:
    - Q(s, a): This represents the estimated value of taking action 'a' in state 's'. This is what the Q-learning algorithm aims to learn.
    - alpha (learning rate): This parameter determines how much new information overrides old information. A high alpha means the agent is more sensitive to recent rewards, while a low alpha means it's more conservative.
    - r (reward): This is the immediate reward received after taking action 'a' in state 's' and transitioning to state 's'.
    - gamma (discount factor): This parameter determines the importance of future rewards. A gamma close to 1 means the agent values future rewards highly, while a gamma close to 0 means it's more myopic and focuses on immediate rewards.
    - s': This is the next state that the agent transitions to after taking action 'a' in state 's'.
    - max_a'(Q(s', a')) : This represents the maximum Q-value that can be obtained from the next state 's' by taking any possible action 'a''. This is the core of the Q-learning update, as it incorporates the estimated future value.
    - [r + gamma * max_a'(Q(s', a')) - Q(s, a)]: This entire term is the Temporal Difference (TD) error. It represents the difference between the estimated value of the current state-action pair (Q(s, a)) and a more accurate estimate of that value based on the immediate reward and the estimated future reward.

    The recurrence relation essentially updates the Q-value of a state-action pair by moving it slightly towards a more accurate estimate derived from the observed reward and the estimated value of the subsequent state.

    - The recurrence relation for value iteration is:

    $V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V_k(s')]$

    This equation describes how the optimal value function $V(s)$ is iteratively approximated. In each iteration $k+1$, the value of a state $s$, denoted by $V_{k+1}(s)$, is updated by considering all possible actions $a$ that can be taken from that state. For each action, it calculates the expected future reward by summing over all possible next states $s'$. This expected future reward is composed of the immediate reward $R(s,a,s')$ received by transitioning to $s'$ and the discounted future value of that next state, $\gamma V_k(s')$, where $V_k(s')$ is the estimated value of state $s'$ from the previous iteration. The $\max_a$ operator then selects the action $a$ that maximizes this expected future reward, thus defining the updated value for state $s$ in the next iteration.
    - The recurrence relation for value iteration is:
        $V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V_k(s')]$
    - This equation calculates the value of a state in the next iteration (k+1) by considering the maximum expected future reward achievable from that state, taking into account the transition probabilities (P), immediate rewards (R), and the discounted values of the next states (V_k) from the current iteration.
- Given value from every state in iteration k, calculate the k+1 iteraon's values for all states:
- In order to recover optimal pllicy, choosing the action to optimize value for the state


- Q value iteration finds sucessive q values, useful for reading off the policy,
  - Initialize with q value 0 for all state all action
  - Given Qk, calculate the k+1 iterations q values for all q states
  - Gemini give the recurrece relation again, and explain it
    - Gemini response:
    - Gemini fix the markdown formatting for this reccurence relation:Q_{k+1}(s,a) = \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma max_{a'} Q_k(s', a')].
      - Gemini response:
        Q_{k+1}(s,a) = \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma \max_{a'} Q_k(s', a')]
    - This relation updates the Q-value for a state-action pair in the next iteration (k+1) by considering the expected future reward, which includes the immediate reward plus the discounted maximum Q-value achievable from the next state (s') over all possible next actions (a').

- Gemini explain hwo they differ in depth and the key differences and why/intuition
  - Gemini response:
    Value iteration and Q-value iteration both aim to find optimal policies in Markov Decision Processes (MDPs) by iteratively updating state-value functions or state-action value functions. Value iteration directly computes the optimal value function V*(s) for each state, and the policy is then derived by choosing the action that maximizes the expected next state value. Q-value iteration, on the other hand, directly computes the optimal action-value function Q*(s,a), from which the optimal policy can be readily read off by selecting the action that yields the maximum Q-value for a given state. The key difference lies in what is being iteratively updated: V(s) in value iteration versus Q(s,a) in Q-value iteration. Q-value iteration is often preferred when the transition and reward functions are unknown (as in reinforcement learning) because it learns the value of taking specific actions in specific states, making policy extraction more direct.


## Q-Learning

- Sample-based Q-value iteration; learn $Q(s,a)$ values as you go.
- Receive samples of $(s, a, s', r)$, similar to TD learning.
- Consider your old estimate.
- Consider the new sample estimate, which is the reward plus the maximum value from following the policy. This addition of the maximum is not present in TD learning.
  - We do this because we are doing Q-learning, which requires specifying an action $a$.
- We incorporate the new information into a running average, similar to TD learning.
- Whatever action you want to take for that state, the only restriction is that you will only choose whatever action maximizes the Q-value for the successor state.
- This is best assuming you don't know all the parameters. It's best for recovering the optimal policy.
- Converges to the optimal policy even if you're acting suboptimally.
  - This is due to estimating the Q-value of every state-action combination.
- Some conditions:
  - There have to be enough samples.
    - Need to explore enough / visit state-action pairs an infinite number of times (Law of Large Numbers).
    - Having an optimal learning rate that is not 1 or 0, otherwise, we won't learn anything.
    - Eventually, make the learning rate small enough so we converge to an estimated value.
      - The more samples you collect, the more you would like to gradually reduce the learning rate.
- No theoretical guarantee or guidance, but this is how you tune your parameters.
  - Can collect a batch of samples, weigh the batch together, etc.

## Reinforcement Learning Framework

- We figured out choosing an action, how about choosing parameters?
- Explore vs. exploit
- Exploration: epsilon-greedy and exploration functions
- Approximate Q-learning
- Every time step:
  - With small probability $\epsilon$, act randomly.
  - With large probability $1 - \epsilon$, act on the current policy.
- You are exploring randomly, which can be slow and take a long time to visit unexplored but important states.
- You eventually explore the space, but keep moving around once learning is done.
  - Can stop this by setting a large $T$ and you will act optimally; lower $\epsilon$ over time, basically.
- Exploration functions are another solution:
  - We take some small information from that state and then move from there.
  - Some sort of informant to further influence where we go.


## Exploration Functions
- Visit states that are less explored
- Find ways to get you to take actions that can lead you to less explored states
- Can utilize an exploration function $f$ that takes a VALUE ESTIMATE and a VISIT COUNT and returns the value with some bonus.
- Inflating Q-value so that you can explore it, where $f(Q,N) = Q + c/N$ where $c$ is some constant.
- Less visited states will enjoy a larger bonus.
- Will propagate to early states that lead to unknown states as well.
  - Not only just getting a one-time bonus, but propagating the bonus back so that you will choose that action to explore.
- Pacman example
  - Instead of a table with all states and actions, utilize similar states.
  - Using things like symmetry in order to map the same problem for different scenarios.
  - Modeling the problem by the position of Pacman and the ghosts would cause a huge state space.
  - e.g., if Pacman is blocked, it leads to a failed state; you can know not to get into a blocked area.
  - Leads to approximate Q-learning.

## Generalizing across states via Approximated Q-learning
- Basic Q-learning involves keeping a table of all Q-values.
- Too many states to visit.
- We want to generalize learning about some smaller number of training states from experience.
- Feature-based representation.
- No longer describing a state using a vector of features, but properties to describe the current state.
- Features are functions from states to real numbers or 0/1 that capture important properties of the state.
  - Maybe relative position of ghosts compared to Pacman?
  - Distance to a dot, Manhattan distance to all the dots.
  - Is Pacman in a tunnel?
  - Can make a feature vector from here.
- Describe a Q-state $(s,a)$ with features.
  - e.g., action moves towards a dot / moving towards a ghost.
  - Binary representation of movement.
  - Similar to heuristic functions in search algorithms.
  - Features can be good or bad, but we can learn the weights of each to know which are good and bad.
- Describe a state using feature-based representation.
- $V(s) = w_1f_1(s) + w_2f_2(s) + ... + w_nf_n(s)$
- $Q(s,a) = w_1f_1(s,a) + w_2f_2(s,a) + ... + w_nf_n(s,a)$
- Learn the weights such that they are telling you something useful for the value of the state.
- The advantage of this is that our experience is summarized in a few powerful numbers/features.
  - Deep learning can be a way to automatically extract what the useful features are.
  - Domain knowledge would help; otherwise, we could accidentally pick an irrelevant feature which will not give us any gain or insights.

- Q learning with linear Qq functions
  - Observe via transitions
  - Calculate the difference
  - Update


## Least squares regression
- Gemini give the equation for least squares regression
  - Gemini response:
    The equation for least squares regression is:

    $$ \hat{\beta} = (X^T X)^{-1} X^T y $$

- Can be used in gradient descent
  - Gemini can you give mroe info on why gradient descrent works and what are some pitfalls we might face whilst using this?
    - Gemini response:
      Gradient descent works by iteratively adjusting the weights of a model in the direction that most rapidly decreases the cost function (or error). It calculates the gradient of the cost function with respect to the weights and takes a step in the negative gradient direction. This process is repeated until the minimum of the cost function is reached. 

      Pitfalls of gradient descent can include: 

      1.  **Local Minima:** The algorithm might get stuck in a local minimum instead of finding the global minimum of the cost function. 
      2.  **Saddle Points:** Similar to local minima, saddle points can also trap the optimization process. 
      3.  **Learning Rate Selection:** Choosing an appropriate learning rate is crucial. A learning rate that is too high can cause the algorithm to overshoot the minimum, while a learning rate that is too low can lead to very slow convergence. 
      4.  **Vanishing/Exploding Gradients:** In deep neural networks, gradients can become extremely small (vanishing) or extremely large (exploding), hindering effective learning. 
      5.  **Feature Scaling:** If features have vastly different scales, gradient descent can converge much slower. 
      6.  **Choice of Batch Size (for stochastic/mini-batch gradient descent):** The batch size impacts the trade-off between computational efficiency and the stability of convergence.

## Q learning
- Least squares minimizes error
- Online least square
- Imgine only one point x with features f(x) target value y and weights y
- Gemini explain how approximate q benefites from least squares and gradient descent
  - Gemini response:
    Approximate Q-learning benefits from least squares and gradient descent in the following ways:

    *   **Least Squares:** In Q-learning, we often approximate the Q-function (which estimates the expected future reward for a given state-action pair) using a linear model or a more complex function approximator. Least squares regression provides a way to find the optimal weights for this approximator by minimizing the squared error between the predicted Q-value and the target Q-value (derived from the Bellman equation). Essentially, least squares helps in finding the best fit for the Q-function to the observed data.
    *   **Gradient Descent:** When the Q-function is represented by a non-linear model (like a neural network), gradient descent is used to optimize the weights of this model. It iteratively adjusts the weights to minimize the error between the predicted Q-values and the target Q-values, allowing the Q-learning agent to learn a more accurate approximation of the optimal Q-function over time.


## MIDTERM IS UP TO THIS LECTURE