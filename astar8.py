#Authors: Owen Brock, Serina Oswalt

import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set

State = Tuple[int, ...]  # length 9, row-major, 0 = blank

MOVES = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

@dataclass(order=True)
class PrioritizedItem:
    f: int
    h: int
    g: int
    state: State = field(compare=False)
    parent: Optional[State] = field(compare=False, default=None)
    move: Optional[str] = field(compare=False, default=None)

def to_state(grid: List[List[int]]) -> State:
    flat = tuple(num for row in grid for num in row)
    if len(flat) != 9:
        raise ValueError("State must have exactly 9 numbers (3x3).")
    return flat

def print_state(state: State) -> None:
    for r in range(3):
        row = state[r*3:(r+1)*3]
        print(" ".join(str(x) if x != 0 else "0" for x in row))
    print()

def find_blank(state: State) -> Tuple[int, int]:
    idx = state.index(0)
    return divmod(idx, 3)  # row and col

def neighbors(state: State) -> List[Tuple[State, str]]:
    br, bc = find_blank(state)
    out = []
    for m, (dr, dc) in MOVES.items():
        nr, nc = br + dr, bc + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            new_state = list(state)
            b_idx = br * 3 + bc
            n_idx = nr * 3 + nc
            new_state[b_idx], new_state[n_idx] = new_state[n_idx], new_state[b_idx]
            out.append((tuple(new_state), m))
    return out

def misplaced_tiles(state: State, goal: State) -> int:
    # Count tiles that are out of place excluding blank
    return sum(1 for i in range(9) if state[i] != 0 and state[i] != goal[i])

def manhattan_distance(state: State, goal: State) -> int:
    # Precompute the goal positions
    goal_pos = {}
    for i, tile in enumerate(goal):
        goal_pos[tile] = (i // 3, i % 3)

    dist = 0
    for i, tile in enumerate(state):
        if tile == 0:
            continue
        r, c = i // 3, i % 3
        gr, gc = goal_pos[tile]
        dist += abs(r - gr) + abs(c - gc)
    return dist

def inversion_parity(state: State) -> int:
    """Parity (even=0, odd=1) of inversion count ignoring 0."""
    arr = [x for x in state if x != 0]
    inv = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    return inv % 2

def is_solvable(start: State, goal: State) -> bool:
    """
    For 3x3 (odd width), solvable iff start and goal have same inversion parity.
    (This works for arbitrary goal configurations, not only 1..8,0.)
    """
    return inversion_parity(start) == inversion_parity(goal)

def reconstruct_path(
    came_from: Dict[State, Tuple[Optional[State], Optional[str]]],
    start: State,
    goal: State
) -> Tuple[List[State], List[str]]:
    states = []
    moves = []
    cur = goal
    while cur != start:
        parent, move = came_from[cur]
        if parent is None or move is None:
            break
        states.append(cur)
        moves.append(move)
        cur = parent
    states.append(start)
    states.reverse()
    moves.reverse()
    return states, moves

def a_star(
    start: State,
    goal: State,
    heuristic_name: str
) -> Dict:
    if heuristic_name not in ("misplaced", "manhattan"):
        raise ValueError("heuristic_name must be 'misplaced' or 'manhattan'.")

    if start == goal:
        return {
            "found": True,
            "path_states": [start],
            "path_moves": [],
            "cost": 0,
            "generated": 1,
            "expanded": 0,
        }

    if not is_solvable(start, goal):
        return {
            "found": False,
            "reason": "Unsolvable (parity mismatch).",
            "generated": 0,
            "expanded": 0,
        }

    def h(s: State) -> int:
        return misplaced_tiles(s, goal) if heuristic_name == "misplaced" else manhattan_distance(s, goal)

    open_heap: List[PrioritizedItem] = []
    came_from: Dict[State, Tuple[Optional[State], Optional[str]]] = {}

    g_score: Dict[State, int] = {start: 0}
    start_h = h(start)
    heapq.heappush(open_heap, PrioritizedItem(start_h, start_h, 0, start, None, None))
    came_from[start] = (None, None)

    closed: Set[State] = set()

    generated = 1   # start node generated
    expanded = 0

    while open_heap:
        current_item = heapq.heappop(open_heap)
        current = current_item.state

        # skip stale entries 
        if current_item.g != g_score.get(current, float("inf")):
            continue

        if current in closed:
            continue

        closed.add(current)
        expanded += 1

        if current == goal:
            path_states, path_moves = reconstruct_path(came_from, start, goal)
            return {
                "found": True,
                "path_states": path_states,
                "path_moves": path_moves,
                "cost": g_score[goal],
                "generated": generated,
                "expanded": expanded,
            }

        for nxt, mv in neighbors(current):
            tentative_g = g_score[current] + 1

            # If we've already finalized nxt with a better g, skip
            if nxt in closed and tentative_g >= g_score.get(nxt, float("inf")):
                continue

            if tentative_g < g_score.get(nxt, float("inf")):
                g_score[nxt] = tentative_g
                came_from[nxt] = (current, mv)
                nxt_h = h(nxt)
                nxt_f = tentative_g + nxt_h
                heapq.heappush(open_heap, PrioritizedItem(nxt_f, nxt_h, tentative_g, nxt, current, mv))
                generated += 1

    return {
        "found": False,
        "reason": "Search exhausted (should be rare if solvable).",
        "generated": generated,
        "expanded": expanded,
    }

def read_grid_from_user(label: str) -> State:
    print(f"Enter {label} state as 3 lines of 3 integers (use 0 for blank).")
    grid = []
    for i in range(3):
        row = input(f"Row {i+1}: ").strip().split()
        if len(row) != 3:
            raise ValueError("Each row must have exactly 3 integers.")
        grid.append([int(x) for x in row])
    flat = [x for r in grid for x in r]
    if sorted(flat) != list(range(9)):
        raise ValueError("State must contain each number 0..8 exactly once.")
    return to_state(grid)

def main():
    print("8-Puzzle Solver (A*)")
    print("Heuristics: misplaced / manhattan")
    print()

    start = read_grid_from_user("INITIAL")
    goal = read_grid_from_user("GOAL")

    print("\nChoose heuristic: (1) misplaced  (2) manhattan")
    choice = input("Your choice [1/2]: ").strip()
    heuristic = "misplaced" if choice == "1" else "manhattan"

    result = a_star(start, goal, heuristic)

    print("\n===== RESULT =====")
    print(f"Heuristic: {heuristic}")
    print(f"Nodes generated: {result.get('generated', 0)}")
    print(f"Nodes expanded:  {result.get('expanded', 0)}")

    if not result["found"]:
        print(f"No solution found. Reason: {result.get('reason', 'Unknown')}")
        return

    print(f"Solution cost (moves): {result['cost']}")
    print(f"Moves: {' '.join(result['path_moves']) if result['path_moves'] else '(already at goal)'}")
    print("\nPath states:")
    for idx, s in enumerate(result["path_states"]):
        print(f"Step {idx}:")
        print_state(s)

if __name__ == "__main__":
    main()
