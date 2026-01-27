# Lecture 2

### Search Problems

*   Uniform Search Methods
    *   DFS (Depth-First Search)
    *   BFS (Breadth-First Search)
    *   Uniform Cost Search
    *   Other Variants
*   Reflex Agents
    *   Choose action based on current percept
    *   May have a model of the world's current state or memory
    *   Do not consider the future consequences of their actions
    *   Consider how the world is
*   Planning Agents
    *   Ask "What if?": make decisions based on consequences of actions
    *   Must have a model of how the world evolves in response to actions
    *   Must formulate a predefined goal
    *   Consider how the world would be
Gemini answer these in bullet points:
*   How do they differ?
    *   Reflex agents act purely on the immediate percept, without foresight or memory of past states beyond what influences the current percept. They are reactive.
    *   Planning agents, in contrast, can reason about the future. They possess a model of how the world changes and can simulate the outcomes of different actions to achieve a goal. They are proactive and deliberative.
    *   Uniform search methods are algorithms used to explore a state space, typically in the context of finding a path from a start state to a goal state. They are the "how" of navigating through possibilities, while reflex and planning agents are the "who" or "what" makes the decisions.
*   Examples of each?
    *   **Reflex Agents:**
        *   A simple thermostat: It turns on the heat when the temperature drops below a set point and turns off when it reaches the set point. It only reacts to the current temperature.
        *   A vacuum cleaner robot that has sensors to detect obstacles and avoids them. It reacts to its immediate surroundings.
    *   **Planning Agents:**
        *   A GPS navigation system: It considers traffic, road closures, and destination to plan the best route. It anticipates future travel time and conditions.
        *   A chess-playing AI: It considers multiple moves ahead, evaluates potential outcomes, and plans a sequence of actions to achieve checkmate.
    *   **Uniform Search Methods:**
        *   **DFS:** Exploring a maze by always taking the first available path until a dead end is reached, then backtracking.
        *   **BFS:** Finding the shortest path in a grid-like structure by exploring all immediate neighbors, then their neighbors, and so on, layer by layer.
        *   **Uniform Cost Search:** Finding the cheapest path in a weighted graph (e.g., finding the least expensive travel route between cities).
*   Can they be used in conjunction?
    *   Yes, they can and often are used in conjunction.
        *   A planning agent might use uniform search methods (like BFS or Uniform Cost Search) to determine the optimal sequence of actions to achieve its goals, given its model of the world.
        *   Even reflex agents can be seen as a very simple form of decision-making where the "planning" horizon is zero, and the action is solely dictated by the current percept. More sophisticated reflex agents might incorporate simple memory, which could be considered a very rudimentary form of state modeling.


*   **Can a reflex agent be rational?**
    *   For reflex agents, the definition of rationality needs to be considered. A rational agent is one that acts to achieve the best outcome, or at least the best expected outcome, given what it knows.
        *   A simple reflex agent [This type of agent acts solely based on the current percept, ignoring past experiences or future consequences.] acts based on a set of condition-action rules.
        *   If these rules are perfectly designed to achieve the desired outcome for every possible situation, then the agent can be considered rational. However, this is often not achievable in practice.
        *   If the rules are incomplete or do not cover all possible situations optimally, the agent will not be rational.
        *   Therefore, a reflex agent *can* be rational if its rule set perfectly maps percepts to optimal actions for all possible states.
*   **Optimal vs. complete planning for planning agents**
    *   **Optimal Planning:** Aims to find the *best* possible sequence of actions to achieve a goal.
        *   This considers not just *if* a goal can be reached, but also the cost or desirability of the path taken.
        *   For example, finding the shortest path, the least costly path, or the fastest path.
        *   Optimal planning is often computationally more expensive than simply finding *any* plan.
    *   **Complete Planning:** Aims to find *any* sequence of actions that will achieve the goal, if one exists.
        *   It does not necessarily find the best plan, but rather a sufficient plan.
        *   This is typically less computationally demanding than optimal planning.
        *   In many scenarios, finding a complete plan is sufficient for the agent to function effectively.
*   **Planning vs. Replanning for planning agents**
    *   **Planning:** The process of generating a sequence of actions to achieve a goal *before* execution begins.
        *   This involves reasoning about the environment and predicting the consequences of actions.
        *   A plan is typically created in a simulated or internal model of the world.
        *   The assumption is often that the world will behave as predicted during execution.
    *   **Replanning:** The process of generating a new plan or modifying an existing plan *during* execution, typically because the environment has changed unexpectedly or the agent has encountered an unforeseen situation.
        *   This is crucial for agents operating in dynamic or uncertain environments where initial plans may become invalid.
        *   Replanning allows the agent to adapt to new information or changing circumstances.
        *   The decision to replan is often triggered by a discrepancy between the predicted outcome of an action and the actual outcome, or by the detection of a new obstacle or opportunity.
        *   For example, if a robot is following a planned path and a new wall appears, it will need to replan its route.



## Pacman Search Problem

- Leads to the Pacman search problem.
- Need a world state that includes every detail of the environment.
- State keeps only the details needed for search (abstraction).
- Example:
  - Problem: Pathing
    - states: (x,y) locations
    - actions: NSEW (North, South, East, West)
    - successor: Updated location
    - goal test: Is (x,y) = END?
  - Problem: Eat all dots
    - {(x,y), dot booleans} [A state represented by Pacman's current (x,y) coordinates and a boolean indicating if each dot in the maze has been eaten.]
    - actions: NSEW
    - successor: Updated location and dot booleans
    - goal test: Dots are all false

- Even with a small world, could have lots of world states. impossible to store all

- Important to handle


## Representation via State Space Graphs
- Nodes are abstracted world configurations
- Each arc represent the action and cost
- The goal test is a set of goal nodes, can be singular or multiple
- In each state space graph, nodes are unique and occur only once
- Rarely build this fullg raph, but can be useful
- Can use a Search tree, where it acts like a what if tree
- Children correspond to successors
- Nodes whow states with routes corresponding to plans that acheive those states
- For most problems you can't build the whole tree
## General Tree Search
- Main question: What are the nodes to explore next?
- In general, search algorithms need to have:
  - Completeness: Is the algorithm guaranteed to find a solution when one exists?
  - Optimality: Does the strategy find the optimal solution?
  - Time complexity: How long does it take to find a solution?
  - Space complexity: How much memory is needed to perform the search?
## Ways to define complexity
- Can define complexity via search algorithm properties
- $m$ tiers of depth
- Branching factors, max # of successors any node can have
- Number of nodes in tree would be $b^m + 1$ (for root) and $= O(b^m)$
## DFS Variant - Depth Limited Search
- DFS with a predetermined depth limit, avoids DFS failure in loops.
- How to choose the depth limit L?
- How many nodes are in the state space graph?
- The maximum number of steps to reach one node from any other node; the diameter of the state space, d.
- Four criteria analysis.

## DFS Iterative Deepening DFS
- Choose the right depth limit L.
- Iteratively increase limit L during the search.

## BFS Variant
- Bidirectional search
  - Starts forward from the initial state and backward from the goal state until the two searches converge.

BFS and DFS are not always efficient because they can explore a large number of nodes that are not on the optimal path, especially in graphs with many branches or deep paths. Uniform Cost Search (UCS), also known as Dijkstra's algorithm, finds the least-cost path by always expanding the node with the lowest path cost from the start. This greedy approach ensures that when a goal node is first reached, it is via the least-cost path.

| Algorithm            | Completeness | Optimality | Time Complexity       | Space Complexity      |
|----------------------|--------------|------------|-----------------------|-----------------------|
| Breadth-First Search (BFS) | Yes          | Yes        | $O(b^d)$              | $O(b^d)$              |
| Depth-First Search (DFS) | No (can be, with cycle detection) | No         | $O(b^m)$              | $O(bm)$               |
| Uniform Cost Search (UCS) | Yes          | Yes        | $O(b^{\lceil C^*/\epsilon \rceil})$ [Note 1] | $O(b^{\lceil C^*/\epsilon \rceil})$ [Note 1] |

[Note 1]: Time and space complexity for UCS are often expressed in terms of $C^*$ (the cost of the optimal solution) and $\epsilon$ (the smallest path cost increment). If all edge costs are at least 1, it simplifies to $O(b^d)$ in the worst case, similar to BFS, but it guarantees optimality.

**Example:**

Let's consider a simple graph:
`A -> B (cost 1)`
`A -> C (cost 3)`
`B -> D (cost 2)`
`C -> D (cost 1)`
Goal: Reach D from A.

**BFS:**
- Explores level by level.
- Path: A -> B -> D (cost 1 + 2 = 3)
- Path: A -> C -> D (cost 3 + 1 = 4)
- BFS would find A -> B -> D first.

**DFS:**
- Explores as deep as possible.
- Path: A -> B -> D (cost 3)
- Path: A -> C -> D (cost 4)
- DFS might find A -> B -> D or A -> C -> D depending on exploration order.

**UCS:**
- Expands nodes with the lowest path cost.
- 1. Expand A (cost 0) -> explore B (cost 1), C (cost 3)
- 2. Expand B (cost 1) -> explore D (cost 1+2=3)
- 3. Expand C (cost 3) -> explore D (cost 3+1=4)
- UCS will expand B before C because B has a lower path cost (1 vs 3). It will find the path A -> B -> D with cost 3.

**Bidirectional Search:**
- Starts from A and D simultaneously.
- Forward search from A: A (cost 0) -> B (cost 1), C (cost 3)
- Backward search from D: D (cost 0) -> B (cost 2), C (cost 1)
- If forward search reaches B (cost 1) and backward search reaches B (cost 2 from D), total cost = 1 + 2 = 3.
- If forward search reaches C (cost 3) and backward search reaches C (cost 1 from D), total cost = 3 + 1 = 4.
- Bidirectional search can significantly reduce the search space by meeting in the middle. The effective search depth is halved, leading to a much lower complexity. For example, if both forward and backward searches explore up to depth d/2, the complexity becomes $O(b^{d/2} + b^{d/2}) = O(b^{d/2})$.