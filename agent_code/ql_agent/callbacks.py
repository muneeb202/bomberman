import os
import pickle
import random
from collections import deque
import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DIRECTIONS = [('UP', 0, -1), ('RIGHT', 1, 0), ('DOWN', 0, 1), ('LEFT', -1, 0)]
FEATURE_DIM = 26
STALL_WATCH_STEPS = 6
INFO_EVERY = 20


def setup(self):
    shape = (2, len(ACTIONS), FEATURE_DIM)
    if self.train or not os.path.isfile('my-saved-model.pt'):
        self.model = np.zeros(shape, dtype=np.float32)
    else:
        with open('my-saved-model.pt', 'rb') as f:
            self.model = pickle.load(f)
        if self.model.shape != shape:
            self.logger.warning('Incompatible model shape %s; starting fresh.', self.model.shape)
            self.model = np.zeros(shape, dtype=np.float32)

    self._stall_watch = deque(maxlen=STALL_WATCH_STEPS)
    self._position_history = deque(maxlen=6)
    self._last_action = None
    self._info_last_step = -INFO_EVERY


def act(self, game_state):
    epsilon = getattr(self, 'epsilon', 0.1)
    valid = valid_action_mask(game_state)
    _, _, _, pos = game_state['self']
    self._stall_watch.append(pos)
    self._position_history.append(pos)

    coin_dist = nearest_coin_distance(game_state)
    crate_dist = nearest_crate_distance(game_state)
    coin_dir = nearest_coin_direction(game_state)
    hits_crate = bomb_would_hit_crate(game_state['field'], pos)

    # Once a coin is reachable, collecting it is the primary objective.
    if coin_dist is not None and coin_dir is not None:
        ci = ACTIONS.index(coin_dir)
        if valid[ci]:
            if game_state['step'] - self._info_last_step >= INFO_EVERY:
                self.logger.info('step=%s pos=%s coin_dist=%s crate_dist=%s -> %s (coin)', game_state['step'], pos, coin_dist, crate_dist, coin_dir)
                self._info_last_step = game_state['step']
            self._last_action = coin_dir
            self._stall_watch.clear()
            return coin_dir

    # If no coin is reachable, safely open a useful crate.
    if valid[5] and hits_crate:
        self.logger.info('step=%s pos=%s coin_dist=%s crate_dist=%s -> BOMB (useful safe crate)', game_state['step'], pos, coin_dist, crate_dist)
        self._last_action = 'BOMB'
        self._stall_watch.clear()
        return 'BOMB'

    # Break stationary WAIT loops.
    if len(self._stall_watch) == STALL_WATCH_STEPS and len(set(self._stall_watch)) == 1:
        moves = [a for a, ok in zip(ACTIONS[:4], valid[:4]) if ok]
        if moves:
            choice = random.choice(moves)
            self.logger.warning('step=%s pos=%s stationary loop -> forcing %s', game_state['step'], pos, choice)
            self._stall_watch.clear()
            self._last_action = choice
            return choice

    cycle = (len(self._position_history) >= 4 and
             self._position_history[-1] == self._position_history[-3] and
             self._position_history[-2] == self._position_history[-4])
    reverse = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}.get(self._last_action)

    if self.train and random.random() < epsilon:
        legal = [a for a, ok in zip(ACTIONS, valid) if ok]
        if legal:
            weights = []
            for a in legal:
                if a == 'BOMB' and hits_crate: weights.append(0.35)
                elif a in ACTIONS[:4]: weights.append(0.20)
                else: weights.append(0.05)
            weights = np.asarray(weights, dtype=float)
            weights /= weights.sum()
            choice = np.random.choice(legal, p=weights)
            self._last_action = choice
            if game_state['step'] - self._info_last_step >= INFO_EVERY:
                self.logger.info('step=%s pos=%s epsilon=%.3f coin=%s crate=%s -> %s (explore)', game_state['step'], pos, epsilon, coin_dist, crate_dist, choice)
                self._info_last_step = game_state['step']
            return choice

    q = (self.model[0] + self.model[1]) @ state_to_features(game_state) / 2.0
    q = np.where(valid, q, -np.inf)

    if coin_dist is not None:
        q[5] = -np.inf
    if any(valid[:4]):
        q[4] -= 0.35
    if reverse and sum(valid[:4]) > 1:
        ri = ACTIONS.index(reverse)
        if valid[ri]: q[ri] -= 1.0
    if valid[5] and hits_crate:
        q[5] += 1.5

    if cycle:
        if reverse and valid[ACTIONS.index(reverse)]:
            q[ACTIONS.index(reverse)] = -np.inf
        self.logger.warning('step=%s pos=%s A-B-A-B cycle detected; avoiding %s', game_state['step'], pos, reverse)

    best = np.flatnonzero(q == np.max(q))
    choice = ACTIONS[int(np.random.choice(best))] if len(best) else 'WAIT'

    if game_state['step'] - self._info_last_step >= INFO_EVERY:
        qstr = ', '.join(f'{a}={v:.2f}' if np.isfinite(v) else f'{a}=X' for a, v in zip(ACTIONS, q))
        self.logger.info('step=%s pos=%s coin=%s crate=%s -> %s | %s', game_state['step'], pos, coin_dist, crate_dist, choice, qstr)
        self._info_last_step = game_state['step']

    self._last_action = choice
    return choice


def valid_action_mask(game_state):
    field, bombs, others = game_state['field'], game_state['bombs'], game_state['others']
    _, _, bomb_possible, (x, y) = game_state['self']
    mask = [is_free(field, bombs, others, x + dx, y + dy) for _, dx, dy in DIRECTIONS]
    mask += [True, bomb_possible and is_safe_to_bomb(field, bombs, others, (x, y))]
    return np.asarray(mask, dtype=bool)


def is_free(field, bombs, others, x, y):
    w, h = field.shape
    if not (0 <= x < w and 0 <= y < h) or field[x, y] != 0: return False
    if any((bx, by) == (x, y) for (bx, by), _ in bombs): return False
    return not any((ox, oy) == (x, y) for _, _, _, (ox, oy) in others)


def bfs(field, bombs, others, start, danger=None):
    danger = danger or set()
    dist, first = {start: 0}, {start: None}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for name, dx, dy in DIRECTIONS:
            p = (x + dx, y + dy)
            if p in dist or not is_free(field, bombs, others, *p) or p in danger: continue
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
                if not (0 <= x < w and 0 <= y < h) or field[x, y] == -1: break
                danger.add((x, y))
                if field[x, y] == 1: break
    xs, ys = np.where(explosion_map > 0)
    danger.update(zip(xs.tolist(), ys.tolist()))
    return danger


def bomb_would_hit_crate(field, pos):
    w, h = field.shape
    x0, y0 = pos
    for _, dx, dy in DIRECTIONS:
        for step in range(1, 4):
            x, y = x0 + dx * step, y0 + dy * step
            if not (0 <= x < w and 0 <= y < h) or field[x, y] == -1: break
            if field[x, y] == 1: return True
    return False


def bomb_would_hit_opponents(field, pos, others):
    w, h = field.shape
    targets = {p for _, _, _, p in others}
    x0, y0 = pos
    for _, dx, dy in DIRECTIONS:
        for step in range(1, 4):
            x, y = x0 + dx * step, y0 + dy * step
            if not (0 <= x < w and 0 <= y < h) or field[x, y] == -1: break
            if (x, y) in targets: return True
            if field[x, y] == 1: break
    return False


def bfs_distance_to_safe(field, bombs, others, danger, start):
    if start not in danger: return 0
    dist, queue = {start: 0}, deque([start])
    while queue:
        x, y = queue.popleft()
        if (x, y) != start and (x, y) not in danger: return dist[(x, y)]
        for _, dx, dy in DIRECTIONS:
            p = (x + dx, y + dy)
            if p in dist or not is_free(field, bombs, others, *p): continue
            dist[p] = dist[(x, y)] + 1
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
    return first_step[min(reachable, key=dist.get)] if reachable else None


def crate_adjacent_tiles(field):
    w, h = field.shape; result = set()
    xs, ys = np.where(field == 1)
    for cx, cy in zip(xs.tolist(), ys.tolist()):
        for _, dx, dy in DIRECTIONS:
            x, y = cx + dx, cy + dy
            if 0 <= x < w and 0 <= y < h and field[x, y] == 0: result.add((x, y))
    return result


def opponent_adjacent_tiles(field, others):
    w, h = field.shape; result = set()
    for _, _, _, (ox, oy) in others:
        for _, dx, dy in DIRECTIONS:
            x, y = ox + dx, oy + dy
            if 0 <= x < w and 0 <= y < h and field[x, y] == 0: result.add((x, y))
    return result


def nearest_coin_direction(game_state):
    if not game_state['coins']: return None
    field, bombs, others = game_state['field'], game_state['bombs'], game_state['others']
    pos = game_state['self'][3]
    danger = compute_danger_tiles(field, bombs, game_state['explosion_map'])
    dist, first = bfs(field, bombs, others, pos, danger)
    return nearest_target_direction(dist, first, game_state['coins'])


def nearest_coin_distance(game_state):
    if not game_state['coins']: return None
    field, bombs, others = game_state['field'], game_state['bombs'], game_state['others']
    pos = game_state['self'][3]
    danger = compute_danger_tiles(field, bombs, game_state['explosion_map'])
    dist, _ = bfs(field, bombs, others, pos, danger)
    vals = [dist[c] for c in game_state['coins'] if c in dist]
    return min(vals) if vals else None


def nearest_crate_distance(game_state):
    field, bombs, others = game_state['field'], game_state['bombs'], game_state['others']
    pos = game_state['self'][3]
    targets = crate_adjacent_tiles(field)
    if not targets: return None
    danger = compute_danger_tiles(field, bombs, game_state['explosion_map'])
    dist, _ = bfs(field, bombs, others, pos, danger)
    vals = [dist[t] for t in targets if t in dist]
    return min(vals) if vals else None


def nearest_opponent_distance(game_state):
    if not game_state['others']: return None
    field, bombs, others = game_state['field'], game_state['bombs'], game_state['others']
    pos = game_state['self'][3]
    targets = opponent_adjacent_tiles(field, others)
    if not targets: return None
    danger = compute_danger_tiles(field, bombs, game_state['explosion_map'])
    dist, _ = bfs(field, bombs, others, pos, danger)
    vals = [dist[t] for t in targets if t in dist]
    return min(vals) if vals else None


def bfs_escape(field, bombs, others, danger, start):
    if start not in danger: return None
    dist, first = {start: 0}, {start: None}; queue = deque([start])
    while queue:
        x, y = queue.popleft()
        if (x, y) != start and (x, y) not in danger: return first[(x, y)]
        for name, dx, dy in DIRECTIONS:
            p = (x + dx, y + dy)
            if p in dist or not is_free(field, bombs, others, *p): continue
            dist[p] = dist[(x, y)] + 1; first[p] = first[(x, y)] or name; queue.append(p)
    return None


def state_to_features(game_state):
    if game_state is None: return None
    field, bombs, others, coins = (game_state[k] for k in ('field', 'bombs', 'others', 'coins'))
    pos = game_state['self'][3]; bomb_possible = game_state['self'][2]
    danger = compute_danger_tiles(field, bombs, game_state['explosion_map'])
    dist, first = bfs(field, bombs, others, pos, danger)
    coin_dir = direction_onehot(nearest_target_direction(dist, first, coins))
    free = [int(is_free(field, bombs, others, pos[0] + dx, pos[1] + dy)) for _, dx, dy in DIRECTIONS]
    escape = direction_onehot(bfs_escape(field, bombs, others, danger, pos))
    crates = crate_adjacent_tiles(field)
    crate_dir = direction_onehot(nearest_target_direction(dist, first, crates))
    opponents = opponent_adjacent_tiles(field, others)
    opponent_dir = direction_onehot(nearest_target_direction(dist, first, opponents))
    return np.asarray([1.0] + coin_dir + free + [int(pos in danger)] + escape + [int(bomb_possible)] + crate_dir + [int(bomb_would_hit_crate(field, pos))] + opponent_dir + [int(bomb_would_hit_opponents(field, pos, others)), int(bomb_possible and is_safe_to_bomb(field, bombs, others, pos))], dtype=np.float32)
