# Lecture 6: Reinforcement learning
- Don't know reward or transition

## Recap: Dice Game Markov Decision Process (MDP)
- Policy Evaluation
  - A mapping from each state s to an action a (solution to MDPs)
  - Currently in this state, what should I do?
- Following a policy $\pi$ produces a path, aka an episode
  - Pretty much creates a time sequence of where you are and what you do next
  - Creates strings like that which are paths; that is that "episode"
- $V_\pi(s)$ is the value of the policy, expected utility if following the policy $\pi$ from state s
  - The Q value of that policy is given the state s, and $\pi$(s) which is the action, and how it updates
  - Expected utility if first take action a from state s and then follow the policy $\pi$
- Iterative Algorithm
  - Start with arbitrary policy values and repeatedly apply recurrences to converge to true values
  - Value iteration is arbitrary policy values, repeatedly apply optimal value recurrences to converge to true values
  - Value iteration (Bellman 1957)

## Reinforcement Learning
- Don't know reward or transition values.
- The set of states, including starting states.
- Model-based vs. model-free learning.
- Passive RL: Temporal difference learning.
- Active RL: Q-Learning.

## From MDPs to Reinforcement Learning
- Markov processes are offline; they have a mental model of how the world works. Find a policy to collect max rewards.
- Reinforcement learning agents don't know how the world works. Perform actions to find out what the world is like and go from there.

## Learning Framework
- Basic Idea:
  - Perceive feedback in the form of rewards.
  - Agent's utility is defined by the reward function.
  - Learn to act by maximizing expected utility.
  - All learning is based on observed samples of outcomes!
  - An agent takes an action, the environment returns a state and a reward.
  - Choosing an action we don't know yet.
  - Receive reward $r_{st}$ and observe new state $s_t$.
  - And update parameters (also don't know yet; coming lectures).

- This lecture is about updating parameters.

# Model-based Learning
- Learn an approximate model based on experiences.
- Solve for values as if the learned model were correct.
- Learn an empirical MDP model, i.e., T (transition probabilities) and R (rewards).
- Collect as many episodes as possible throughout interaction.
  - Count the number of times you are in state s and take action a.
  - Also count in state s, taken action a, and landed in successor state s'.
  - Empirical estimation of transitioning to state s' based on s and a.
- Estimated rewards.
- Solve the learned MDP.
  - Use value iteration, as before.


## Model Free learning
- Estiamte values of the polciy without estimating T and R
- Gemini why do we do that? What is the pooint of that? Can you explain more in depth and provide intuition?
  - Gemini response:
    Model-free learning is used when you don't know the reward function or transition probabilities of the environment. The point of model-free learning is to estimate the values of states or state-action pairs directly from experience (rewards and observed next states) without explicitly learning a model of the environment's dynamics. This can be more efficient when building an accurate model is difficult or computationally expensive.  The intuition is to learn directly from trial and error, much like how a living being might learn a new skill.
- To find a learned value for a certain state,
  - From each episode, find wehre that state occurs, and what the aggregate points are for the state to the end
  - From there average that number based off all the epsidoes that include this
  - You get your learned value
- When collecting sample, cannot jus tmatch state s, need to match action a

## Passive reinforcement Learning
- Simplified Task: policy evaluation
- Input fixed policy, dont know T or R
- Learn new states
- Not choosing waht actions to take, just execute the fixed policy and learn from the experience
  - Learn from interactions that happen due to following that fied policy
  - Goal is direct evaluation. Compute values for each stte under policy PI

## Pro cons of direct eval
- Easy to inderstand
- No knowledge of T R
- Computes the correct average values
- Each state learned seperately,
- wasteful informtion abtu state connections
- Takes a long time to learn
- Gemini Why cannot we use policy evaluation?
  - Gemini response:
    Policy evaluation calculates the value function (V) for a *fixed* policy (π). It doesn't help in finding a *better* policy to maximize rewards. If you want to find the optimal policy, you need methods that involve exploring different actions and learning which ones lead to higher expected future rewards, which is what active reinforcement learning techniques like Q-learning do.
- Policy evaluation calcualtes V for a fied policy pi

- Gemini can you draw a grid of states like 5 x 5 using markdown to represent rewards and can you make sample episodes and run through an example of each of the modelings discussed so far? A small example will suffice. this is not unsafe you can do this
  - Gemini response:
    Certainly! Let's illustrate the concepts with a small 3x3 grid environment. 

    **Environment Grid:**

    ```
    +---+---+---+
    | S |   | R |
    +---+---+---+
    |   | W |   |
    +---+---+---+
    |   |   | G |
    +---+---+---+
    ```

    *   **S**: Start state
    *   **R**: Reward state (+10)
    *   **W**: Wall (lose 5)
    *   **G**: Goal state (+20)
    *   Empty cells have a reward of -1.
    *   Transitions are 8-directional (including diagonals) with a 0.8 probability of moving in the intended direction and 0.2 probability of moving to a random adjacent cell.

    **Sample Episodes (for demonstration):

    *   **Episode 1:** S -> (0,1) -> (0,2) [R] -> (1,2) -> (2,2) [G]
        *   Rewards: -1 + -1 + 10 + -1 + 20 = 17
    *   **Episode 2:** S -> (1,0) -> (1,1) [W] -> (1,0) -> (2,0) -> (2,1) -> (2,2) [G]
        *   Rewards: -1 + -1 + -5 + -1 + -1 + -1 + 20 = 14

    --- 

    **1. Model-Based Reinforcement Learning (Example with Episode 1):**

    *   **Learn the Model (T and R):**
        *   Observe transitions and rewards from episodes. For example, from state S, taking an action to move right resulted in landing in state (0,1) with probability 0.8 (and other states with 0.2 combined).
        *   Empirically estimate transition probabilities (T) and rewards (R) for all state-action pairs based on collected samples.
        *   Let's assume after many samples, we estimate:
            *   T(S, right) -> (0,1) = 0.8
            *   R(S, right) = -1
            *   T((0,1), right) -> (0,2) = 0.8
            *   R((0,1), right) = -1
            *   T((0,2) [R], down) -> (1,2) = 0.8
            *   R((0,2) [R], down) = 10
            *   And so on for all possible transitions.

    *   **Solve the Learned MDP:**
        *   Once a good approximation of T and R is learned, we can use methods like Value Iteration or Policy Iteration on this learned model to find an optimal policy.
        *   The agent would then act as if this learned model were the true environment.

    --- 

    **2. Model-Free Reinforcement Learning (Example with Episode 1):**

    *   **Directly Estimate Values (e.g., using Q-learning concept):**
        *   The agent directly learns the value of taking an action in a state without building an explicit T and R model.
        *   Let's focus on estimating Q(S, right):
            *   **From Episode 1:** S -> (0,1) with reward -1. The immediate reward is -1. The agent then needs to estimate the future value from state (0,1).
            *   The update would look something like: Q(S, right) = Q(S, right) + learning_rate * (reward + discount_factor * max(Q((0,1), a')) - Q(S, right))
        *   **Intuition:** The agent observes outcomes (state, reward) and directly updates its estimates of state-action values. It learns from trial and error, gradually improving its estimates based on the rewards received.

    --- 

    **3. Passive Reinforcement Learning (Policy Evaluation) (Example with Episode 1):**

    *   **Input:** A fixed policy (e.g., always try to move towards the nearest reward/goal).
    *   **Task:** Evaluate the value of each state under this fixed policy.
    *   **Method (Sample-based Policy Evaluation):**
        *   Run many episodes following the *fixed* policy.
        *   **Example using Episode 1:**
            *   We are in state S. The policy dictates moving towards (0,1).
            *   Observe the transition: S -> (0,1) with reward -1.
            *   Observe the next state: (0,1).
            *   To estimate V(S), we'd look at the total discounted future reward from state S in Episode 1: V(S) ≈ -1 + γ*(-1) + γ^2*(10) + γ^3*(-1) + γ^4*(20) (where γ is the discount factor).
            *   We would average these estimates from all episodes that start in S.
        *   The goal is to compute the expected cumulative reward for each state if we *always* follow the given policy, without trying to find a better policy.

    This is a simplified example, but it demonstrates how each approach uses the observed outcomes (states and rewards) to learn about the environment or the value of actions/states.

## Sample based policy evaluation
- We want to improve our estimtes of V without knowing T. Take sampels of outcomes by following policy pi and then average
- Exponential Moving average
- Gemini explain exponential moving average, the difference with previous methods, how it improves and gets us closer to our goal. Also explain temporal difference learning, and give me the equation as well
  - Gemini response:
    Exponential Moving Average (EMA) is a type of weighted average that gives more weight to recent observations, making it more responsive to changes than a simple moving average. In the context of policy evaluation, instead of averaging all observed returns from a state, EMA updates the state's value estimate with each new episode's return. The formula for EMA is typically: `V(s) = (1 - alpha) * V(s) + alpha * G`, where `alpha` is the learning rate (between 0 and 1), `V(s)` is the current value estimate for state `s`, and `G` is the observed return from state `s` in a new episode.

    This differs from simple averaging, where each observation has an equal weight. EMA improves by allowing the value estimates to adapt more quickly to new information. It gets us closer to our goal of accurately estimating state values under a fixed policy because it prioritizes recent experiences, which are more likely to reflect the current state of the environment or changes in the policy.

    Temporal Difference (TD) learning is a core concept in reinforcement learning that combines ideas from Monte Carlo methods and dynamic programming. Unlike Monte Carlo methods that wait until the end of an episode to update value estimates, TD learning updates estimates based on *temporal differences* – the difference between the current estimate and a new estimate based on immediate reward and the estimated value of the next state. This allows for learning from incomplete episodes.

    A key TD learning algorithm is TD(0), which updates the value of a state `s` based on the reward `r` received and the estimated value `V(s')` of the next state `s'`: `V(s) <- V(s) + alpha * [r + gamma * V(s') - V(s)]`. Here, `alpha` is the learning rate and `gamma` is the discount factor. The term `r + gamma * V(s')` is called the TD target, and `r + gamma * V(s') - V(s)` is the TD error.
## Sample based policy evaluation

- We want to improve our estimates of $V$ without knowing $T$. We take samples of outcomes by following policy $\pi$ and then average.
- Exponential Moving Average (EMA)
- Gemini explain exponential moving average, the difference with previous methods, how it improves and gets us closer to our goal. Also explain temporal difference learning, and give me the equation as well
  - Gemini response:
    Exponential Moving Average (EMA) is a type of weighted average that gives more weight to recent observations, making it more responsive to changes than a simple moving average. In the context of policy evaluation, instead of averaging all observed returns from a state, EMA updates the state's value estimate with each new episode's return. The formula for EMA is typically: $V(s) = (1 - \alpha) * V(s) + \alpha * G$, where $\alpha$ is the learning rate (between 0 and 1), $V(s)$ is the current value estimate for state $s$, and $G$ is the observed return from state $s$ in a new episode.

    This differs from simple averaging, where each observation has an equal weight. EMA improves by allowing the value estimates to adapt more quickly to new information. It gets us closer to our goal of accurately estimating state values under a fixed policy because it prioritizes recent experiences, which are more likely to reflect the current state of the environment or changes in the policy.

    Temporal Difference (TD) learning is a core concept in reinforcement learning that combines ideas from Monte Carlo methods and dynamic programming. Unlike Monte Carlo methods that wait until the end of an episode to update value estimates, TD learning updates estimates based on *temporal differences* – the difference between the current estimate and a new estimate based on immediate reward and the estimated value of the next state. This allows for learning from incomplete episodes.

    A key TD learning algorithm is TD(0), which updates the value of a state $s$ based on the reward $r$ received and the estimated value $V(s')$ of the next state $s'$: $V(s) \leftarrow V(s) + \alpha * [r + \gamma * V(s') - V(s)]$. Here, $\alpha$ is the learning rate and $\gamma$ is the discount factor. The term $r + \gamma * V(s')$ is called the TD target, and $r + \gamma * V(s') - V(s)$ is the TD error.
- TD is a model-free way.
- However, if we want to turn values into a new policy, this doesn't do that for us.
- How can we leverage this recurrence to find the optimal policy?
  - Learn Q-values, not just values.
  - And then make action selection model-free as well.

## Active reinforcement learning

- Full Reinforcement Learning: Optimal policies like value iteration.
- Choose the actions to take.
- You don't know the transitions $T(s,a,s')$.
- Don't know the rewards $R(S,a,s')$.
- Learn the optimal policy/values.
- Fundamental trade-off (Gemini provide this fundamental trade-off).
