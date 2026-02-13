# Lecture 8 - Adversial Search

## Recap
- Approx Q learning
- Goal to learn weights of the features
- Intuition is that we adjust the relevance of ative features according to observed samples
- Memoray and generalization: One experience updates many states implicitly. keyword many
- More update per observation

- for policy search, oftentimes feature based plicies work well, action selection priority, get the ordering of q values right
- Start with ok solution and optimize weights from there

## Adversarial Search
- Agent goal would still be to find optimal policy whilst taking into account wha tthe other agents would perform
- Different games
  - Checkers, Chess, Go, Packman
  - Checkers was completely solved. If both players play optimally, we an predict outome.
    - gemini give the hisotry of the Chinook and how it was solved in 2007. Short 1-2 sentences. Also explain why if the other palyer is not playing optimally why it makes it harder
      - Gemini response:
        Chinook, a computer program developed at the University of Alberta, solved the game of Checkers in 2007. This means that with optimal play from both sides, the outcome of a Checkers game is predetermined. If the other player is not playing optimally, it makes the game harder to predict because their suboptimal moves introduce unpredictability and create scenarios that weren't accounted for in the optimal strategy.
    - Alpha beta search
  - Chess, alpha beta search over 30 B positions with depths of 14, 14 moves ahead
    - Singular extensions up to depth 20
    - 8000 Features, 4000 opening book moves, all endgames with 5 peices
    - 700,000 grandmaster games
    - Null move heuristic
      - Opponent gets to move twice (what if i do nothing and oponnent moves)
- Many types of adversarial games
  - Deterministic or stochastic
  - Simultaenous or sequential
  - Number of players
  - Zero Sum
    - Gemini give some examples
      - Gemini response:
        Examples of adversarial games include Checkers, Chess, Go, and Pac-Man. These games involve multiple players with competing goals, where each agent aims to find an optimal policy while considering the actions of their opponents. Deterministic games like Checkers and Chess contrast with stochastic games, and some, like Chess, utilize techniques such as alpha-beta search to explore vast numbers of possible game states.
  - Perfect information


## Deterministic games
- Many formalizations
- States, players, actions, transition function S x A ? S
- Terminal test, S -> T,F
- Terminal utilities: S X P -> R
- Many are zero sum, agents have opposite utilities, my win opponents loss
- A Single value that one maximizes and the other minimizes
- General games exist whcih agents have independent utiliy, cooperatio competition or a mixture of both
- States can be positions of all peices and whose move, the actions can be legal chess moves, and the end state is whether the state is in checkmate or draw. The utility can be if a player wins or loses
- Value of a state would be the bet achievable outcome/utility from that state. Only terminal rewards
  - Utilize a recurrence relationship, starting with terminal states with known values and finding the values for non-terminal states.
  - gemini can you make a simple example with a pacman and one dot and draw out a small tree to visualize this using ASCII characters for markdown?
    - Gemini response:
      Okay, here's a simple Pac-Man example with one dot and a small tree visualized using ASCII characters.

      Imagine a 2x2 grid. Pac-Man (P) starts at the bottom left (0,0). There's one dot (D) at the top right (1,1). The goal for Pac-Man is to reach the dot.

      We can represent the state by (Pacman_X, Pacman_Y, Dot_Collected_Status).

      Let's assume Pac-Man can move Up (U) or Right (R).

      **Tree Visualization:**

      ```
      (0,0,False) --R--> (1,0,False)
           |                |
           |                |--R--> (2,0,False) [Out of Bounds - Assume invalid move or ends]
           |                |
           |                |--U--> (1,1,False)
           |                              |
           |                              |--U--> (1,2,False) [Out of Bounds]
           |                              |
           |                              |--R--> (2,1,False) [Out of Bounds]
           |
           |--U--> (0,1,False)
                    |
                    |--R--> (1,1,False)
                             |
                             |--U--> (1,2,False) [Out of Bounds]
                             |
                             |--R--> (2,1,False) [Out of Bounds]
                    |
                    |--U--> (0,2,False) [Out of Bounds]

      ```

      **Explanation:**

      *   **Nodes:** Each node represents a state (Pacman's position, whether the dot is collected).
      *   **Edges:** Arrows represent Pac-Man's actions (R for Right, U for Up).
      *   **Starting State:** (0,0,False) - Pac-Man at bottom left, dot not collected.
      *   **Goal State:** When Pac-Man reaches the coordinate of the dot (1,1), the `Dot_Collected_Status` becomes `True`.  For simplicity here, we're just showing Pac-Man reaching the dot's location. A full game would have utilities associated with reaching the dot.
      *   **Terminal States:** In this very simplified example, reaching the dot's location is the primary objective. States where Pac-Man moves out of bounds could be considered terminal with a very low utility (a loss).

      In a real adversarial search scenario, you'd have opponent moves and utilities for winning/losing/drawing. Here, it's a single agent problem for simplicity to illustrate the state tree.

- Now to generalize this to an adversarial Game trees
- Minimax values
  - Simplify the game as definiing thevalue of th state as the egents value
  - All terminal states are known
  - A state space search tree, where each layer is either max or min
  - Players alternate turns
  - ompute each node's minimax value which would be the best aheivable utility againt a rational/optimal adversary
- For simultaneous mvoe games, would need a game matrix
- If the state is a terminal state, return the states utility, if the next agent is max, return max state if next is min return in state
- We can intialize eah sate as + or - infinity
- We can use DFS to impliment minimax
- Gemini expand on using DFS to do minimax, give pseudocode and generic python implementation
  - Gemini response:
    Depth-First Search (DFS) is a natural fit for implementing the Minimax algorithm because it explores one branch of the game tree as far as possible before backtracking. This aligns with how Minimax recursively calculates the value of states.

    **Pseudocode for Minimax with DFS:**

    ```
    function minimax(state, depth, maximizing_player):
        if depth == 0 or is_terminal(state):
            return utility(state)

        if maximizing_player:
            max_eval = -infinity
            for each child_state in generate_children(state):
                eval = minimax(child_state, depth - 1, False)
                max_eval = max(max_eval, eval)
            return max_eval
        else:  // minimizing_player
            min_eval = +infinity
            for each child_state in generate_children(state):
                eval = minimax(child_state, depth - 1, True)
                min_eval = min(min_eval, eval)
            return min_eval

    // Initial call:
    // minimax(initial_state, max_depth, True)  (assuming the first player is the maximizer)
    ```

    **Generic Python Implementation:**

    ```python
    import math

    def minimax(state, depth, is_maximizing_player, generate_children, is_terminal, get_utility):
        if depth == 0 or is_terminal(state):
            return get_utility(state)

        if is_maximizing_player:
            max_eval = -math.inf
            for child_state in generate_children(state):
                eval = minimax(child_state, depth - 1, False, generate_children, is_terminal, get_utility)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = math.inf
            for child_state in generate_children(state):
                eval = minimax(child_state, depth - 1, True, generate_children, is_terminal, get_utility)
                min_eval = min(min_eval, eval)
            return min_eval

    # Example Usage (you would need to define these functions for a specific game):
    def sample_generate_children(state):
        # Returns a list of possible next states from the current state
        pass

    def sample_is_terminal(state):
        # Returns True if the state is a terminal state (game over), False otherwise
        pass

    def sample_get_utility(state):
        # Returns the utility value of a terminal state
        pass

    # initial_state = ...
    # max_depth = ...
    # result = minimax(initial_state, max_depth, True, sample_generate_children, sample_is_terminal, sample_get_utility)
    ```

    **Explanation of the Python Code:**

    *   **`minimax(state, depth, is_maximizing_player, ...)`**: This is the core recursive function.
        *   `state`: The current game state.
        *   `depth`: The remaining depth to search. When it reaches 0, we stop searching down that path and return the utility of the current state.
        *   `is_maximizing_player`: A boolean flag indicating whether the current turn belongs to the maximizing player (True) or the minimizing player (False).
        *   `generate_children`: A function that takes a `state` and returns an iterable of all possible successor `child_state`s.
        *   `is_terminal`: A function that takes a `state` and returns `True` if it's a terminal state, `False` otherwise.
        *   `get_utility`: A function that takes a terminal `state` and returns its numerical utility value.

    *   **Base Case**: If `depth` is 0 or the `state` is terminal, the function returns the `utility` of that state.

    *   **Maximizing Player**: If `is_maximizing_player` is `True`, the function iterates through all `child_state`s, recursively calls `minimax` for each with `is_maximizing_player` set to `False` (since it's the opponent's turn), and keeps track of the maximum `eval` found. It initializes `max_eval` to negative infinity.

    *   **Minimizing Player**: If `is_maximizing_player` is `False`, it does the opposite: iterates through children, calls `minimax` with `is_maximizing_player` set to `True`, and finds the minimum `eval`. It initializes `min_eval` to positive infinity.

    This DFS-based approach systematically explores the game tree, assigning the best possible outcome for the current player at each node, assuming the opponent also plays optimally.


## Minimax efficiency
- How efficient is minimax for time and space complexity?
  - Just like DFS, exauhstive
  - Time is b*m where b is the branching factor and m is the depth/#moves
  - Space is O(bm)
  - Chess the branching is 35 and m is about 100. Exact solution is completely infeasible, too time complex for computers
  - Cannot explore full tree


## Resource Limits
- Problem iN realisti games, cannot search to leaves
- SOlution, Depth limited search
- Only search to a limited depth in the tree
- use an evaluation function for nn terminal positons
- gemini give an example of eval funtions for pacman or chess with brevity
  - Gemini response:
    For Pac-Man, an evaluation function could prioritize states where Pac-Man is close to a dot, has power pellets available, and is far from ghosts. For Chess, it might consider material advantage (pieces on the board), king safety, pawn structure, and piece mobility.

- Guarantee of optimal play is gone, but use of iterative deepening for anytime algoirthm
- Search depth matters, eval functions are always imperfect
- the deeper in the tree the eval is buried, the relatively easier we can learn a high quality eval function
  - The tradeoff between complexity of features and complexity of computation
  - The deeper, the more states + the cloeer we are to terminal states. We expand more ndoes than previous levels
- Do we need to fully expand the whole tree
- We won't look at terminal values via minimax pruning. Whenever we see terminal node values that are polar in a certain unoptimal direction, we can get rid of some computation
  - Alpha beta pruning