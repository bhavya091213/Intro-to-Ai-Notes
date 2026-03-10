from __future__ import annotations

import heapq
from collections import deque
from typing import Dict, List, Tuple, Optional


# -----------------------------
# Graph + Heuristic
# -----------------------------
GRAPH: Dict[str, Dict[str, int]] = {
    "S": {"A": 2, "B": 5},
    "A": {"C": 3, "F": 6},
    "B": {"C": 2, "E": 4},
    "C": {"D": 9, "E": 7},
    "D": {"G": 2},
    "E": {"G": 3},
    "F": {"G": 1},
    "G": {},
}

H: Dict[str, int] = {
    "S": 7,
    "A": 5,
    "B": 6,
    "C": 4,
    "D": 3,
    "E": 2,
    "F": 5,
    "G": 0,  # standard assumption
}


# -----------------------------
# Shared helpers
# -----------------------------
def format_path(path: Optional[List[str]]) -> str:
    return " -> ".join(path) if path else "No path"


def edge_cost(u: str, v: str) -> int:
    return GRAPH[u][v]


# "States expanded" convention used here:
# Count a state when it is removed from the frontier and we iterate over its neighbors.
# Do NOT count the goal as expanded (we stop immediately when we pop it).
def bfs(start: str, goal: str) -> Tuple[Optional[List[str]], Optional[int], int]:
    q = deque([(start, [start], 0)])
    visited = {start}
    expanded = 0

    while q:
        node, path, g = q.popleft()
        if node == goal:
            return path, g, expanded

        expanded += 1
        for nbr in sorted(GRAPH[node]):  # alphabetical expansion
            if nbr in visited:
                continue
            visited.add(nbr)
            q.append((nbr, path + [nbr], g + edge_cost(node, nbr)))

    return None, None, expanded


def dfs(start: str, goal: str) -> Tuple[Optional[List[str]], Optional[int], int]:
    stack = [(start, [start], 0)]
    visited = {start}
    expanded = 0

    while stack:
        node, path, g = stack.pop()
        if node == goal:
            return path, g, expanded

        expanded += 1
        # reverse-sort so that LIFO produces alphabetical traversal
        for nbr in sorted(GRAPH[node], reverse=True):
            if nbr in visited:
                continue
            visited.add(nbr)
            stack.append((nbr, path + [nbr], g + edge_cost(node, nbr)))

    return None, None, expanded


def ucs(start: str, goal: str) -> Tuple[Optional[List[str]], Optional[int], int]:
    # Tie-break by node name; store best_g to handle re-discoveries cleanly.
    heap: List[Tuple[int, str, List[str]]] = [(0, start, [start])]
    best_g: Dict[str, int] = {start: 0}
    expanded = 0

    while heap:
        g, node, path = heapq.heappop(heap)

        # skip stale entries
        if g != best_g.get(node, float("inf")):
            continue

        if node == goal:
            return path, g, expanded

        expanded += 1
        for nbr in sorted(GRAPH[node]):  # alphabetical generation
            new_g = g + edge_cost(node, nbr)
            if new_g < best_g.get(nbr, float("inf")):
                best_g[nbr] = new_g
                heapq.heappush(heap, (new_g, nbr, path + [nbr]))

    return None, None, expanded


def greedy(start: str, goal: str) -> Tuple[Optional[List[str]], Optional[int], int]:
    # Priority is h(n). Tie-break by node name.
    heap: List[Tuple[int, str, List[str], int]] = [(H[start], start, [start], 0)]
    visited = set()
    expanded = 0

    while heap:
        _, node, path, g = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return path, g, expanded

        expanded += 1
        for nbr in sorted(GRAPH[node]):
            if nbr in visited:
                continue
            heapq.heappush(heap, (H[nbr], nbr, path + [nbr], g + edge_cost(node, nbr)))

    return None, None, expanded


def astar(start: str, goal: str) -> Tuple[Optional[List[str]], Optional[int], int]:
    # Priority is f = g + h. Tie-break by node name.
    heap: List[Tuple[int, str, List[str], int]] = [(H[start], start, [start], 0)]
    best_g: Dict[str, int] = {start: 0}
    expanded = 0

    while heap:
        f, node, path, g = heapq.heappop(heap)

        # skip stale entries
        if g != best_g.get(node, float("inf")):
            continue

        if node == goal:
            return path, g, expanded

        expanded += 1
        for nbr in sorted(GRAPH[node]):
            new_g = g + edge_cost(node, nbr)
            if new_g < best_g.get(nbr, float("inf")):
                best_g[nbr] = new_g
                new_f = new_g + H[nbr]
                heapq.heappush(heap, (new_f, nbr, path + [nbr], new_g))

    return None, None, expanded


# -----------------------------
# Runner
# -----------------------------
def main() -> None:
    start, goal = "S", "G"

    algos = [
        ("BFS", bfs),
        ("DFS", dfs),
        ("UCS", ucs),
        ("Greedy", greedy),
        ("A*", astar),
    ]

    expansions_by_algo: Dict[str, int] = {}
    costs_by_algo: Dict[str, Optional[int]] = {}

    for name, func in algos:
        path, cost, expanded = func(start, goal)
        expansions_by_algo[name] = expanded
        costs_by_algo[name] = cost

        print(f"{name} path: {format_path(path)}")
        print(f"{name} cost: {cost}")
        print(f"States Expanded: {expanded}\n")

    # Fastest by expansions
    min_exp = min(expansions_by_algo.values())
    fastest = [k for k, v in expansions_by_algo.items() if v == min_exp]
    print(f"Fewest expansions: {', '.join(fastest)} ({min_exp})")

    # Lowest cost among those that found a path
    valid_costs = {k: v for k, v in costs_by_algo.items() if v is not None}
    if valid_costs:
        min_cost = min(valid_costs.values())
        optimal = [k for k, v in valid_costs.items() if v == min_cost]
        print(f"Lowest path cost: {', '.join(optimal)} ({min_cost})")


if __name__ == "__main__":
    main()
