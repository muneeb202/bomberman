import os
import pickle
import random
from collections import deque

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DIRECTIONS = [('UP', 0, -1), ('RIGHT', 1, 0), ('DOWN', 0, 1), ('LEFT', -1, 0)]

# bias(1) + coin-direction onehot(4) + free-direction flags(4)
# + in_danger(1) + escape-direction onehot(4) + bomb_possible(1)
FEATURE_DIM = 15


def setup(self):
    """
    Setup your code. This is called once when loading each agent.

    The model is a pair of independent linear Q-functions (Double
    Q-learning): Q_A(s, a) = model[0][a] . phi(s), Q_B likewise with
    model[1]. Action selection uses their average; train.py's update_q
    uses one to pick the best next action and the other to evaluate it,
    which prevents the max-bootstrapping overestimation that a single
    Q-function is prone to.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = np.zeros((2, len(ACTIONS), FEATURE_DIM), dtype=np.float32)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Epsilon-greedy policy over the averaged Double Q-function.

    self.epsilon is set/decayed in train.py's setup_training/end_of_round.
    Outside training (or before setup_training has run) we fall back to a
    small fixed value so this still behaves sensibly if called directly.
    """
    epsilon = getattr(self, 'epsilon', 0.1)
    if self.train and random.random() < epsilon:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    features = state_to_features(game_state)
    q_values = (self.model[0] + self.model[1]) @ features / 2.0

    valid = valid_action_mask(game_state)
    q_values = np.where(valid, q_values, -np.inf)

    self.logger.debug(f"Q-values: {q_values}")
    return ACTIONS[int(np.argmax(q_values))]


def valid_action_mask(game_state: dict) -> np.array:
    """Boolean array over ACTIONS marking which are currently legal."""
    field = game_state['field']
    bombs = game_state['bombs']
    others = game_state['others']
    _, _, bomb_possible, (x, y) = game_state['self']

    mask = []
    for _, dx, dy in DIRECTIONS:
        mask.append(is_free(field, bombs, others, x + dx, y + dy))
    mask.append(True)           # WAIT is always legal
    mask.append(bomb_possible)  # BOMB only if own bomb isn't ticking
    return np.array(mask)


def is_free(field: np.array, bombs, others, x: int, y: int) -> bool:
    """Whether tile (x, y) can currently be moved onto."""
    width, height = field.shape
    if not (0 <= x < width and 0 <= y < height):
        return False
    if field[x, y] != 0:
        return False
    if any((bx, by) == (x, y) for (bx, by), _ in bombs):
        return False
    if any((ox, oy) == (x, y) for _, _, _, (ox, oy) in others):
        return False
    return True


def bfs(field: np.array, bombs, others, start: tuple):
    """
    Breadth-first search over free tiles starting at `start`.
    Returns (dist, first_step): shortest distance to, and the first move
    direction toward, every reachable tile.
    """
    dist = {start: 0}
    first_step = {start: None}
    queue = deque([start])
    while queue:
        cx, cy = queue.popleft()
        for name, dx, dy in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) not in dist and is_free(field, bombs, others, nx, ny):
                dist[(nx, ny)] = dist[(cx, cy)] + 1
                first_step[(nx, ny)] = first_step[(cx, cy)] or name
                queue.append((nx, ny))
    return dist, first_step


def nearest_coin_distance(game_state: dict):
    """Shortest-path distance to the nearest coin, or None if none reachable."""
    if game_state is None or not game_state['coins']:
        return None
    field = game_state['field']
    bombs = game_state['bombs']
    others = game_state['others']
    _, _, _, pos = game_state['self']
    dist, _ = bfs(field, bombs, others, pos)
    reachable = [dist[c] for c in game_state['coins'] if c in dist]
    return min(reachable) if reachable else None


def compute_danger_tiles(field: np.array, bombs, explosion_map: np.array) -> set:
    """
    All tiles that are currently, or will imminently be, inside a blast.
    A blast travels up to 3 tiles in each direction from a bomb, stopped
    by stone walls; it also destroys (and is stopped by) the first crate
    it hits. Tiles already on fire (explosion_map > 0) are included too.
    """
    width, height = field.shape
    danger = set()
    for (bx, by), _timer in bombs:
        danger.add((bx, by))
        for _, dx, dy in DIRECTIONS:
            for step in range(1, 4):
                nx, ny = bx + dx * step, by + dy * step
                if not (0 <= nx < width and 0 <= ny < height):
                    break
                if field[nx, ny] == -1:  # stone wall stops the blast
                    break
                danger.add((nx, ny))
                if field[nx, ny] == 1:   # crate stops the blast beyond it
                    break
    xs, ys = np.where(explosion_map > 0)
    danger.update(zip(xs.tolist(), ys.tolist()))
    return danger


def bomb_would_hit_crate(field: np.array, pos: tuple) -> bool:
    """Whether a bomb dropped at `pos` would reach at least one crate."""
    width, height = field.shape
    px, py = pos
    for _, dx, dy in DIRECTIONS:
        for step in range(1, 4):
            nx, ny = px + dx * step, py + dy * step
            if not (0 <= nx < width and 0 <= ny < height):
                break
            if field[nx, ny] == -1:  # stone wall stops the blast
                break
            if field[nx, ny] == 1:   # crate found within range
                return True
    return False


def bfs_escape(field: np.array, bombs, others, danger: set, start: tuple):
    """
    If `start` is inside `danger`, returns the first-step direction toward
    the nearest tile that is NOT in danger. Returns None if not in danger,
    or if no escape route is found.
    """
    if start not in danger:
        return None
    dist = {start: 0}
    first_step = {start: None}
    queue = deque([start])
    while queue:
        cx, cy = queue.popleft()
        if (cx, cy) != start and (cx, cy) not in danger:
            return first_step[(cx, cy)]
        for name, dx, dy in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) not in dist and is_free(field, bombs, others, nx, ny):
                dist[(nx, ny)] = dist[(cx, cy)] + 1
                first_step[(nx, ny)] = first_step[(cx, cy)] or name
                queue.append((nx, ny))
    return None


def state_to_features(game_state: dict) -> np.array:
    """
    Feature vector phi(s):
      [0]     bias term (always 1)
      [1:5]   one-hot direction (UP, RIGHT, DOWN, LEFT) of the shortest path
              to the nearest coin (all zero if none reachable)
      [5:9]   whether each of (UP, RIGHT, DOWN, LEFT) is currently free
      [9]     1 if the agent's current tile is inside a bomb blast (now or
              imminently), else 0
      [10:14] one-hot escape direction toward the nearest safe tile
              (all zero if not in danger, or no escape route found)
      [14]    1 if dropping a bomb is currently legal, else 0
    """
    if game_state is None:
        return None

    field = game_state['field']
    bombs = game_state['bombs']
    others = game_state['others']
    coins = game_state['coins']
    explosion_map = game_state['explosion_map']
    _, _, bomb_possible, pos = game_state['self']

    dist, first_step = bfs(field, bombs, others, pos)

    coin_onehot = [0, 0, 0, 0]
    reachable_coins = [c for c in coins if c in dist]
    if reachable_coins:
        nearest = min(reachable_coins, key=lambda c: dist[c])
        direction = first_step[nearest]
        if direction is not None:
            coin_onehot[[d[0] for d in DIRECTIONS].index(direction)] = 1

    free = [1 if is_free(field, bombs, others, pos[0] + dx, pos[1] + dy) else 0
            for _, dx, dy in DIRECTIONS]

    danger = compute_danger_tiles(field, bombs, explosion_map)
    in_danger = 1 if pos in danger else 0

    escape_onehot = [0, 0, 0, 0]
    escape_dir = bfs_escape(field, bombs, others, danger, pos)
    if escape_dir is not None:
        escape_onehot[[d[0] for d in DIRECTIONS].index(escape_dir)] = 1

    return np.array(
        [1.0] + coin_onehot + free + [in_danger] + escape_onehot + [1 if bomb_possible else 0],
        dtype=np.float32,
    )