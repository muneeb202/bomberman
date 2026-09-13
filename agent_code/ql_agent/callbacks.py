import os
import pickle
import random
from collections import deque
import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DIRECTIONS = [('UP', 0, -1), ('RIGHT', 1, 0), ('DOWN', 0, 1), ('LEFT', -1, 0)]
FEATURE_DIM = 26
STALL_WATCH_STEPS = 6


def setup(self):
    if self.train or not os.path.isfile('my-saved-model.pt'):
        self.model = np.zeros((2, len(ACTIONS), FEATURE_DIM), dtype=np.float32)
    else:
        with open('my-saved-model.pt', 'rb') as f:
            self.model = pickle.load(f)
        if self.model.shape != (2, len(ACTIONS), FEATURE_DIM):
            self.logger.warning('Incompatible model shape %s; starting fresh.', self.model.shape)
            self.model = np.zeros((2, len(ACTIONS), FEATURE_DIM), dtype=np.float32)
    self._stall_watch = deque(maxlen=STALL_WATCH_STEPS)
    self._position_history = deque(maxlen=4)
    self._last_action = None


def act(self, game_state):
    epsilon = getattr(self, 'epsilon', 0.1)
    valid = valid_action_mask(game_state)
    _, _, _, pos = game_state['self']
    self._stall_watch.append(pos)
    self._position_history.append(pos)

    # Task 2: if no coin is currently reachable, immediately use a safe,
    # useful bomb instead of allowing the linear model to wander forever.
    coin_dist = nearest_coin_distance(game_state)
    if valid[5] and coin_dist is None and bomb_would_hit_crate(game_state['field'], pos):
        self._last_action = 'BOMB'
        self._stall_watch.clear()
        return 'BOMB'

    # Break WAIT loops.
    if len(self._stall_watch) == STALL_WATCH_STEPS and len(set(self._stall_watch)) == 1:
        moves = [a for a, ok in zip(ACTIONS[:4], valid[:4]) if ok]
        if moves:
            self._stall_watch.clear()
            choice = random.choice(moves)
            self._last_action = choice
            return choice

    # Epsilon exploration, but bias exploration toward movement and useful bombs.
    if self.train and random.random() < epsilon:
        legal = [a for a, ok in zip(ACTIONS, valid) if ok]
        if not legal:
            return 'WAIT'
        weights = []
        for a in legal:
            if a == 'BOMB' and bomb_would_hit_crate(game_state['field'], pos):
                weights.append(0.30)
            elif a in ACTIONS[:4]:
                weights.append(0.20)
            elif a == 'WAIT':
                weights.append(0.05)
            else:
                weights.append(0.05)
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()
        choice = np.random.choice(legal, p=weights)
        self._last_action = choice
        return choice

    q = (self.model[0] + self.model[1]) @ state_to_features(game_state) / 2.0
    q = np.where(valid, q, -np.inf)

    # WAIT should lose whenever there is another movement option.
    if any(valid[:4]):
        q[4] -= 0.35

    # Never immediately undo the previous move when another route exists.
    reverse = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}.get(self._last_action)
    if reverse and valid[ACTIONS.index(reverse)] and sum(valid[:4]) > 1:
        q[ACTIONS.index(reverse)] = -np.inf

    # Strongly prefer a safe bomb that actually destroys a crate.
    if valid[5] and bomb_would_hit_crate(game_state['field'], pos):
        q[5] += 1.5 if coin_dist is not None else 3.0

    # Detect A-B-A-B cycles and avoid returning to the previous tile.
    if len(self._position_history) == 4:
        p = list(self._position_history)
        if p[0] == p[2] and p[1] == p[3] and p[0] != p[1]:
            if reverse and valid[ACTIONS.index(reverse)]:
                q[ACTIONS.index(reverse)] = -np.inf

    best = np.flatnonzero(q == np.max(q))
    choice = ACTIONS[int(np.random.choice(best))] if len(best) else 'WAIT'
    self._last_action = choice
    return choice


def valid_action_mask(game_state):
    field, bombs, others = game_state['field'], game_state['bombs'], game_state['others']
    _, _, bomb_possible, (x, y) = game_state['self']
    mask = [is_free(field, bombs, others, x + dx, y + dy) for _, dx, dy in DIRECTIONS]
    mask.append(True)
    mask.append(bomb_possible and is_safe_to_bomb(field, bombs, others, (x, y)))
    return np.asarray(mask, dtype=bool)


def is_free(field, bombs, others, x, y):
    w, h = field.shape
    if not (0 <= x < w and 0 <= y < h) or field[x, y] != 0:
        return False
    if any((bx, by) == (x, y) for (bx, by), _ in bombs):
        return False
    return not any((ox, oy) == (x, y) for _, _, _, (ox, oy) in others)


def bfs(field, bombs, others, start, danger=None):
    danger = danger or set()
    dist, first = {start: 0}, {start: None}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for name, dx, dy in DIRECTIONS:
            p = (x + dx, y + dy)
            if p in dist or not is_free(field, bombs, others, *p) or p in danger:
                continue
            dist[p] = dist[(x, y)] + 1
            first[p] = first[(x, y)] or name
            queue.append(p)
    return dist, first


def compute_danger_tiles(field, bombs, explosion_map):
    w, h = field.shape
    danger = set()
    for (bx, by), _ in bombs:
        danger.add((bx, by))
        for _, dx, dy in DIRECTIONS:
            for step in range(1, 4):
                x, y = bx + dx * step, by + dy * step
                if not (0 <= x < w and 0 <= y < h) or field[x, y] == -1:
                    break
                danger.add((x, y))
                if field[x, y] == 1:
                    break
    xs, ys = np.where(explosion_map > 0)
    danger.update(zip(xs.tolist(), ys.tolist()))
    return danger


def bomb_would_hit_crate(field, pos):
    w, h = field.shape
    x0, y0 = pos
    for _, dx, dy in DIRECTIONS:
        for step in range(1, 4):
            x, y = x0 + dx * step, y0 + dy * step
            if not (0 <= x < w and 0 <= y < h) or field[x, y] == -1:
                break
            if field[x, y] == 1:
                return True
    return False


def bomb_would_hit_opponents(field, pos, others):
    w, h = field.shape
    targets = {p for _, _, _, p in others}
    x0, y0 = pos
    for _, dx, dy in DIRECTIONS:
        for step in range(1, 4):
            x, y = x0 + dx * step, y0 + dy * step
            if not (0 <= x < w and 0 <= y < h) or field[x, y] == -1:
                break
            if (x, y) in targets:
                return True
            if field[x, y] == 1:
                break
    return False


def bfs_distance_to_safe(field, bombs, others, danger, start):
    if start not in danger:
        return 0
    dist, queue = {start: 0}, deque([start])
    while queue:
        x, y = queue.popleft()
        if (x, y) != start and (x, y) not in danger:
            return dist[(x, y)]
        for _, dx, dy in DIRECTIONS:
            p = (x + dx, y + dy)
            if p in dist or not is_free(field, bombs, others, *p):
                continue
            dist[p] = dist[(x, y)] + 1
            queue.append(p)
    return None


def bfs_escape(field, bombs, others, danger, start):
    if start not in danger:
        return None
    dist, first, queue = {start: 0}, {start: None}, deque([start])
    while queue:
        x, y = queue.popleft()
        if (x, y) != start and (x, y) not in danger:
            return first[(x, y)]
        for name, dx, dy in DIRECTIONS:
            p = (x + dx, y + dy)
            if p in dist or not is_free(field, bombs, others, *p):
                continue
            dist[p] = dist[(x, y)] + 1
            first[p] = first[(x, y)] or name
            queue.append(p)
    return None


def is_safe_to_bomb(field, bombs, others, pos):
    hypothetical = list(bombs) + [(pos, 3)]
    danger = compute_danger_tiles(field, hypothetical, np.zeros_like(field))
    steps = bfs_distance_to_safe(field, hypothetical, others, danger, pos)
    return steps is not None and steps <= 3


def direction_onehot(direction):
    return [int(direction == name) for name, _, _ in DIRECTIONS]


def nearest_target_direction(dist, first_step, targets):
    reachable = [t for t in targets if t in dist]
    if not reachable:
        return None
    return first_step[min(reachable, key=dist.get)]


def crate_adjacent_tiles(field):
    w, h = field.shape
    result = set()
    xs, ys = np.where(field == 1)
    for cx, cy in zip(xs.tolist(), ys.tolist()):
        for _, dx, dy in DIRECTIONS:
            x, y = cx + dx, cy + dy
            if 0 <= x < w and 0 <= y < h and field[x, y] == 0:
                result.add((x, y))
    return result


def opponent_adjacent_tiles(field, others):
    w, h = field.shape
    result = set()
    for _, _, _, (ox, oy) in others:
        for _, dx, dy in DIRECTIONS:
            x, y = ox + dx, oy + dy
            if 0 <= x < w and 0 <= y < h and field[x, y] == 0:
                result.add((x, y))
    return result


def nearest_coin_distance(game_state):
    if game_state is None or not game_state['coins']:
        return None
    field, bombs, others = game_state['field'], game_state['bombs'], game_state['others']
    _, _, _, pos = game_state['self']
    danger = compute_danger_tiles(field, bombs, game_state['explosion_map'])
    dist, _ = bfs(field, bombs, others, pos, danger)
    vals = [dist[c] for c in game_state['coins'] if c in dist]
    return min(vals) if vals else None


def nearest_crate_distance(game_state):
    field, bombs, others = game_state['field'], game_state['bombs'], game_state['others']
    _, _, _, pos = game_state['self']
    targets = crate_adjacent_tiles(field)
    if not targets:
        return None
    danger = compute_danger_tiles(field, bombs, game_state['explosion_map'])
    dist, _ = bfs(field, bombs, others, pos, danger)
    vals = [dist[t] for t in targets if t in dist]
    return min(vals) if vals else None


def nearest_opponent_distance(game_state):
    if game_state is None or not game_state['others']:
        return None
    field, bombs, others = game_state['field'], game_state['bombs'], game_state['others']
    _, _, _, pos = game_state['self']
    targets = opponent_adjacent_tiles(field, others)
    if not targets:
        return None
    danger = compute_danger_tiles(field, bombs, game_state['explosion_map'])
    dist, _ = bfs(field, bombs, others, pos, danger)
    vals = [dist[t] for t in targets if t in dist]
    return min(vals) if vals else None


def state_to_features(game_state):
    if game_state is None:
        return None
    field, bombs, others, coins = (game_state[k] for k in ('field', 'bombs', 'others', 'coins'))
    _, _, bomb_possible, pos = game_state['self']
    danger = compute_danger_tiles(field, bombs, game_state['explosion_map'])
    dist, first = bfs(field, bombs, others, pos, danger)
    coin_dir = direction_onehot(nearest_target_direction(dist, first, coins))
    free = [int(is_free(field, bombs, others, pos[0] + dx, pos[1] + dy)) for _, dx, dy in DIRECTIONS]
    in_danger = int(pos in danger)
    escape = direction_onehot(bfs_escape(field, bombs, others, danger, pos))
    crates = crate_adjacent_tiles(field)
    crate_dir = direction_onehot(nearest_target_direction(dist, first, crates))
    hits_crate = int(bomb_would_hit_crate(field, pos))
    opponents = opponent_adjacent_tiles(field, others)
    opponent_dir = direction_onehot(nearest_target_direction(dist, first, opponents))
    hits_opponent = int(bomb_would_hit_opponents(field, pos, others))
    safe_bomb = int(bomb_possible and is_safe_to_bomb(field, bombs, others, pos))
    return np.asarray([1.0] + coin_dir + free + [in_danger] + escape + [int(bomb_possible)] + crate_dir + [hits_crate] + opponent_dir + [hits_opponent, safe_bomb], dtype=np.float32)
