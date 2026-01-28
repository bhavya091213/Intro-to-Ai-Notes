# Lectures 2–3 Exam Review (Search + Heuristics)

## 1. MASTER REVIEW SHEET

### Agents: Reflex vs Planning (and where search fits)
- Core idea: reflex = react now; planning = simulate consequences to reach a goal; search = a tool planning uses to pick action sequences.
- Key definitions:
  - Reflex agent: maps current percept → action (condition–action rules), no explicit lookahead.
  - Planning agent: uses a model + goal to evaluate action sequences (“what if?”) before acting.
  - Rational agent: maximizes expected performance given its knowledge.
  - Complete planning ≠ optimal planning: complete finds some plan if one exists; optimal finds the best-cost plan.
  - Planning vs replanning: plan-before-execute vs update plan during execution when the world/model mismatch is detected.
- Rules / algorithms:
  - Reflex can be rational if its rules implement the optimal action for every reachable situation (rare in complex worlds).
  - Replan triggers: unexpected obstacles, new info, or observed outcome ≠ predicted outcome.
- Intuition (why it works): planning adds lookahead; reflex is “0-step horizon”.
- ⚠ Common exam traps / misconceptions:
  - ⚠ Confusing “search algorithm” (procedure) with “agent” (decision-maker).
  - ⚠ Assuming reflex agents are automatically irrational.
  - ⚠ Mixing up complete vs optimal (complete does not mean cheapest).

### Problem formulation + representations (state space graphs vs search trees)
- Core idea: you can’t search until you define a problem as states + transitions + goals (+ costs).
- Key definitions:
  - State: minimal information needed to predict future outcomes from actions (abstraction).
  - Initial state, actions, successor function, goal test, step cost / path cost.
  - State space graph: each distinct state appears once (conceptual model).
  - Search tree: nodes are paths/plans; the same state can appear multiple times via different paths.
- Rules / algorithms:
  - “Good” state representation: includes everything needed for goal/successor tests, excludes irrelevant detail → smaller search.
  - If the goal depends on history (e.g., “eat all dots”), encode that history in the state (e.g., remaining-dot indicators).
- Intuition (why it works): abstraction shrinks the space while keeping correctness (goal + dynamics still computable).
- ⚠ Common exam traps / misconceptions:
  - ⚠ Confusing a world state (all details) with a search state (only what matters for decisions).
  - ⚠ Forgetting to encode goal-relevant info (leads to “can’t tell if solved” or wrong successors).
  - ⚠ Underestimating state explosion (small worlds can still have huge state spaces).

### Uninformed search (BFS / DFS / UCS + key variants)
- Core idea: expand nodes using only the problem definition (no “closeness-to-goal” signal).
- Key definitions (compressed):
  - Frontier (open): generated, not yet expanded; Explored (closed): expanded states (for graph search).
  - Branching factor `b`, shallowest goal depth `d`, maximum depth `m`, optimal cost `C*`, minimum step cost `ε`.
- Rules / algorithms:
  - BFS: expand smallest depth first (FIFO queue) → shortest path in steps when all step costs equal.
  - DFS: expand deepest first (LIFO stack) → low memory, can dive into dead ends.
  - Depth-limited DFS: DFS with cutoff `L` (prevents infinite descent).
  - Iterative deepening DFS (IDDFS): run depth-limited DFS with `L = 0,1,2,...` until goal.
  - UCS (Dijkstra): expand lowest path cost `g(n)` first (priority queue) → least-cost path with positive step costs.
  - Bidirectional (usually BFS): search forward from start + backward from goal; meet in middle when actions are reversible and goal is explicit.
- Intuition (why it works): BFS/UCS are “systematic” about guarantees (depth/cost ordering); DFS trades guarantees for memory.
- ⚠ Common exam traps / misconceptions:
  - ⚠ “BFS is optimal” is only true for equal step costs (otherwise use UCS).
  - ⚠ Returning as soon as you *generate* a goal is wrong for UCS (and A*); return when the goal is *popped/dequeued* as best.
  - ⚠ DFS can be incomplete in infinite-depth spaces or with cycles (without proper cycle checks).

### Complexity + evaluation criteria (what you’re usually asked to compare)
- Core idea: algorithms are graded on guarantees (complete/optimal) and resources (time/space) as functions of `b,d,m`.
- Key definitions:
  - Completeness: finds a solution if one exists.
  - Optimality: finds a lowest-cost solution (or shallowest if unit costs).
  - Time/space complexity: nodes expanded / memory stored (often exponential in `d`).
- Rules / formulas:
  - Tree size up to depth `m`: `1 + b + b^2 + ... + b^m = O(b^m)`.
  - BFS: time `O(b^d)`, space `O(b^d)` (memory is the bottleneck).
  - DFS: time `O(b^m)`, space `O(bm)` (space is linear in depth).
  - UCS (common bound): exponential in `C*/ε` (worst case), but optimal with positive step costs.
  - Bidirectional BFS: about `O(b^(d/2))` (huge reduction when applicable).
- Intuition (why it works): exponential growth comes from branching; halving depth is a massive win.
- ⚠ Common exam traps / misconceptions:
  - ⚠ Ignoring space complexity (BFS/UCS often fail due to memory first).
  - ⚠ Mixing `d` (shallowest goal depth) with `m` (max depth explored).

### Informed search: heuristics + A*
- Core idea: use a heuristic `h(n)` to guide expansion toward the goal; A* balances “cost so far” + “estimated remaining”.
- Key definitions:
  - Heuristic `h(n)`: estimate of remaining cost-to-goal from `n`.
  - Greedy best-first: expand lowest `h(n)` (fast, not cost-aware).
  - A*: expand lowest `f(n) = g(n) + h(n)` where `g(n)` is path cost so far.
  - Admissible (optimistic) heuristic: `0 ≤ h(n) ≤ h*(n)` (never overestimates true remaining cost).
- Rules / algorithms:
  - A* termination rule (exam favorite): with admissible `h` and positive step costs, the first goal *popped* from the priority queue is optimal.
  - Designing admissible heuristics: solve a relaxed version of the problem (ignore constraints like walls) → cheap lower bound.
  - 8-puzzle heuristic ladder:
    - Misplaced tiles: lower bound on moves (admissible, weaker).
    - Manhattan distance: sum of tile distances to goal positions (admissible, typically stronger).
    - “True distance” is perfect but defeats the point (it’s as hard as solving).
  - Combining heuristics: if `h1,h2` admissible, then `max(h1,h2)` is admissible (often expands fewer nodes).
- Intuition (why it works): `g` prevents “getting lured” by low `h` paths that are already expensive; `h` prevents blind exploration.
- ⚠ Common exam traps / misconceptions:
  - ⚠ Greedy best-first ≠ A* (greedy ignores `g`, so it can be very suboptimal).
  - ⚠ “More informed” must still be admissible for A* optimality (overestimates break the guarantee).
  - ⚠ Assuming A* is fast by default; if `h = 0`, A* collapses to UCS.

## 2. ACTIVE RECALL QUESTIONS (no answers)

### Agents: Reflex vs Planning (and where search fits)
- Why can a reflex agent be rational in principle, and why is that hard in practice?
- How does a planning agent use a model differently than a reflex agent?
- When does replanning trigger, and what evidence forces it?
- Compare complete planning vs optimal planning: what guarantee changes?
- How do uniform search methods relate to “planning” (tool vs agent)?
- What information must an agent have to simulate “what if?” outcomes?

### Problem formulation + representations (state space graphs vs search trees)
- What are the core components of a search problem definition?
- Why is “abstraction” necessary, and what breaks if you abstract too aggressively?
- When will the same world configuration appear multiple times in a search tree?
- How does a state space graph differ from a search tree in duplicate states and why does it matter?
- What must be included in the state for “eat all dots” type goals, and why?
- What triggers state explosion even in small grid worlds?

### Uninformed search (BFS / DFS / UCS + key variants)
- Compare BFS vs UCS: when do they produce the same answer, and when do they differ?
- Why is returning upon goal *generation* wrong for UCS?
- When does DFS fail to be complete, and what fix attempts to address it?
- How does IDDFS combine BFS-like completeness with DFS-like space?
- When is bidirectional search applicable, and what assumptions does it need?
- What exactly does UCS prioritize, and how is that different from BFS depth?
- What makes BFS’s space usage so large?

### Complexity + evaluation criteria (what you’re usually asked to compare)
- How do `b`, `d`, and `m` differ, and which ones appear in BFS vs DFS bounds?
- Why does `O(b^d)` show up so often in search?
- Which algorithms tend to be memory-bound first, and why?
- How does bidirectional search change the exponent, and why is that powerful?
- What role do positive step costs play in UCS/A* guarantees?

### Informed search: heuristics + A*
- What is a heuristic, and what does it mean for it to be problem-specific?
- Compare greedy best-first vs A*: what term is missing in greedy, and what guarantee is lost?
- Why does admissibility (`h ≤ h*`) imply A* optimality (high-level argument)?
- What does “solve a relaxed problem” mean, and why does it often produce admissible `h`?
- Which is usually stronger for the 8-puzzle: misplaced tiles or Manhattan distance, and why?
- Why is `max(h1,h2)` still admissible, and when is it useful?
- When does A* behave exactly like UCS?

## 3. RAPID CRAM MODE

### Problem setup
- Search problem = `(states, start, actions, successor, goal test, cost)`
- State = “just enough” info for successors + goal; too much → huge space, too little → wrong/undefined
- State space graph: unique states; Search tree: paths, duplicates allowed

### Guarantees + rules of thumb
- BFS: FIFO by depth → complete; optimal only with equal step costs; time/space `O(b^d)`
- DFS: LIFO → low space `O(bm)`; not optimal; incomplete in infinite/cyclic spaces without checks
- DLS/IDDFS: cutoff `L`; IDDFS = increasing `L` → BFS-like completeness with DFS-like space
- UCS (Dijkstra): PQ by `g(n)` → optimal with positive costs; return when goal is popped
- Greedy best-first: PQ by `h(n)` → fast but not optimal (can chase misleading `h`)
- A*: PQ by `f(n)=g(n)+h(n)` → optimal if `h` admissible; `h=0` ⇒ UCS

### Heuristics (must-know)
- Admissible: `0 ≤ h(n) ≤ h*(n)` (never overestimates)
- Build `h` from relaxed problem (ignore constraints) → cheap lower bound
- 8-puzzle: Manhattan distance usually dominates misplaced tiles (both admissible)
- If `h1,h2` admissible → `max(h1,h2)` admissible (often fewer expansions)
