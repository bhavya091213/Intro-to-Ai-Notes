# Lecture 3

## Informed Search vs uninformed search
- uninformed was last lecture
- states/configuration sof the world
- successor functions/ world dynamics
- Search trees/node trees represent plans for reaching states, which each have costs
- Search algoirthms build search trees, ordering of expansion of unexplored nodes
- Optimal pre defined goal, find least cost plan
- in a state graph, no repetitive state, in a state tree you will have duplicate state
- Expanding strategy, BFS expands ndoe with smalelst steps, DFS is largest steps, UCS expands with minimum costs. Can implement all these with some sort of priority queue
- Search will only be as good as how your model capturing the real world/world
  - not just depenednent on quality of inference algorithm
- Evaluate each algorithm by different criteria
  - Completeness (if there is a solution)
  - Time complexity (How long it takes)
  - Space complexity (space efficient)
  - Optimal solution (dependent on predefined goals, shortest path, minimum cost, etc)



## UCS
- Wastes effort by exploring every direction, no information abt progress,
  - Int he vide, the darker the square, the later in the algorithm that node was expanded
- In general, UCS is complete and optimal
- Uninformed, not taking into account if the new state is better or worse compared to the goal state, no info if you are getting closer to the end state or not


### This lecture will focus on informed search, embedding how good or bad a state is


## Pancake Problem
- Goal test: pancakes stack in ordeer of their size
- Cost number of pancakes flipped 
- In the example in classm If we flip 2, we get closer to goal, flipping 3, gets us to goal, flippinf 4 gets us to upside down, and 1 gets us to the original
  - Can build a state space graph with costs as weight
  - With egeneral tree search, we can expand to possible next states

# Search heuristics
- A function that estimates how close a state is to a goal
- Designed for a particular search problem
- For pathing? for all dots?
  - Distance to a dot? How many dots are left?
  - Can have features about current state that give you an estimate if you are getting closer are further from the goal state
- Greedy search on some heuristic might not find optimal path
  - Not guaranteed optimality
  - Will complete
  - Worst case, badly guided DFS

## A* Algorithm
- Unform cost orders by path cost, or backward cumulative costs g(n)
- greedy orders by goal proximity, or forward cost h(n)
- Using g(n) + h(n) to guide the search
- A* should finish when you dequeue the goal
- Have to expand other nodes as well
- A* doesnt fix everything, need good heuristic


## What is a good heuristic
- Admissibilility
- A heurisitc h is admissibile/optimistic if 0 <= h(n) <= h*(n)
  - h*(n) is true cost to a nearest goal
  - Admissible becasue heuristic is simple and easily calculated solution to a relaxed problem, ie ignoring walls

## Optimality of A*
- W admissible heuristic
- Proof that it works only with admissible heuristic
- Assume A is optimal goal node along with B, but B is a subotpimal goal node
- h is admissible
- We claim that A will exit th PQ before B
- imagine B is on the fronteir, candidate to be explored
- Some ancestor n of A is on frontier
- claim is n will be expanded before B

1. f(n) is less or equal to f(A)
2. f(A) is less than f(B)
3. all of descendants of n will be expanded before B
4. A* search is optimal


## 8 Puzzle

- The goal is:
- _ 1 2
- 3 4 5
- 6 7 8

1. to model as a search problem, need to indetify
   1. what are states?
      1. numbers in each cell
   2. how many?
      1. 9! states
   3. What are the actions?
      1. move nearby tile to empty slot
   4. How many successors from start state?
      1. move tile to empty slot
   5. What should the costs be?
      1. cost of 1 for each tile moved

2. Number of tiles not in the right place?
   1. atleast a cost of 1 to correct a tile, this is optimistic therefore admissible
   2. Drastically reduces the number of nodes expanded, 
3. Manhattan distance from correct place
   1. knows that not in the right place, but how far away? More info, better for more steps
4. The actual distance
   1. Is admissible, upper bound
   2. It is ground truth, therefore it would
   3. The problem is you would need to find the ground truth, which is computationally heavy