"""
SARSA Agent - callbacks.py
==========================
All game-state helpers (BFS, danger detection, feature extraction, action
masking) are shared with ql_agent. The difference is in train.py: SARSA is
on-policy, so the TD target uses Q(s', a') for the action actually taken at
s' rather than max_a' Q(s', a').

Note on ordering: the environment calls act(s_t) and only afterwards calls
game_events_occurred(s_t, a_t, s_t+1). At that moment a_t+1 has not been
chosen yet, so act() cannot hand the next action forward. train.py therefore
holds each transition for one step and applies it once a' is known. Nothing
needs to be cached here.
"""

import os
import pickle
import random
from collections import deque

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DIRECTIONS = [('UP', 0, -1), ('RIGHT', 1, 0), ('DOWN', 0, 1), ('LEFT', -1, 0)]

# Feature vector layout (total = 27):
#   [0]       bias
#   [1..4]    coin direction one-hot  (UP/RIGHT/DOWN/LEFT), 0s if no reachable coin
#   [5..8]    free neighbours         (UP/RIGHT/DOWN/LEFT)
#   [9]       in_danger flag
#   [10..13]  timed escape direction  (UP/RIGHT/DOWN/LEFT), 0s if already safe
#   [14]      bomb_possible
#   [15..18]  nearest crate-adjacent direction
#   [19]      bomb_would_hit_crate
#   [20]      safe_to_bomb
#   [21..24]  nearest opponent direction one-hot (UP/RIGHT/DOWN/LEFT), 0s if no opponent
#   [25]      bomb_would_hit_opponent
#   [26]      nearest opponent distance (normalised by 10)
FEATURE_DIM = 27


# ---------------------------------------------------------------------------
# Agent lifecycle
# ---------------------------------------------------------------------------

def setup(self):
    """
    Initialise the linear weight matrix.

    Model shape: (n_actions, feature_dim) — a single weight matrix W where
    Q(s, a) = W[a] · φ(s).  Unlike Double Q-learning we use one set of
    weights; SARSA's on-policy target already reduces maximisation bias.
    """
    shape = (len(ACTIONS), FEATURE_DIM)
    if self.train or not os.path.isfile('my-saved-model.pt'):
        self.logger.info('Setting up SARSA model from scratch (shape=%s).', shape)
        self.model = np.zeros(shape, dtype=np.float64)
    else:
        self.logger.info('Loading SARSA model from saved state.')
        with open('my-saved-model.pt', 'rb') as f:
            self.model = pickle.load(f)
        if self.model.shape != shape:
            self.logger.warning(
                'Incompatible saved model shape %s; starting fresh.', self.model.shape
            )
            self.model = np.zeros(shape, dtype=np.float64)

    # Stall/cycle watchdog (same as ql_agent - see comment there).
    self._stall_watch = deque(maxlen=8)


def act(self, game_state: dict) -> str:
    """
    Epsilon-greedy policy over the linear Q function.

    The returned action is read back by train.py as the `self_action`
    argument of game_events_occurred, so there is nothing to cache here.
    """
    # Default to 0.0 at eval time (self.epsilon is never set outside training).
    epsilon = getattr(self, 'epsilon', 0.0)
    valid = valid_action_mask(game_state)
    _, _, _, pos = game_state['self']
    step = game_state['step']

    if not hasattr(self, '_stall_watch'):
        self._stall_watch = deque(maxlen=8)
    self._stall_watch.append(pos)

    features = state_to_features(game_state)
    q_vals = self.model @ features          # shape (n_actions,)
    q_masked = np.where(valid, q_vals, np.nan)

    def _q_str():
        parts = [f'{a}:{"off" if np.isnan(v) else f"{v:+.3f}"}' for a, v in zip(ACTIONS, q_masked)]
        return ' | '.join(parts)

    # --- Epsilon-greedy exploration ---
    if self.train and random.random() < epsilon:
        legal = [a for a, ok in zip(ACTIONS, valid) if ok]
        choice = random.choice(legal)
        self.logger.info('[ACT] step=%d pos=%s eps=%.3f EXPLORE -> %s | Q=[%s]',
                         step, pos, epsilon, choice, _q_str())
        return choice

    # --- Stall/cycle watchdog ---
    # Fires when all 8 recent positions come from only 2 distinct tiles.
    #
    # Dead-end problem: if the corridor has exactly 2 walkable tiles (e.g.
    # (1,14)↔(1,15) boxed by walls/crates), every non-BOMB direction leads
    # back into the same 2 tiles — the original "pick any non-WAIT/BOMB move"
    # just re-picks UP or DOWN and the cycle never breaks.
    #
    # Fix: compute which moves lead to a tile OUTSIDE the cycling set.
    # If none exist (true dead-end), allow BOMB as the escape if safe.
    is_stalled = (
        len(self._stall_watch) == self._stall_watch.maxlen
        and len(set(self._stall_watch)) <= 2
    )
    if is_stalled:
        cycling_tiles = set(self._stall_watch)
        dir_delta = {'UP': (0, -1), 'RIGHT': (1, 0), 'DOWN': (0, 1), 'LEFT': (-1, 0)}
        # Moves whose destination is NOT in the cycling set
        escape_moves = [
            a for a, ok in zip(ACTIONS, valid)
            if ok and a in dir_delta
            and (pos[0] + dir_delta[a][0], pos[1] + dir_delta[a][1]) not in cycling_tiles
        ]
        if escape_moves:
            choice = random.choice(escape_moves)
        elif valid[ACTIONS.index('BOMB')] and is_safe_to_bomb(
            game_state['field'], game_state['bombs'],
            game_state['others'], pos, game_state['explosion_map'],
        ):
            # True dead-end: bomb the blocking crate wall to open a new path
            choice = 'BOMB'
        else:
            # Last resort: any non-WAIT valid move
            fallback = [a for a, ok in zip(ACTIONS, valid) if ok and a != 'WAIT']
            choice = random.choice(fallback) if fallback else 'WAIT'
        self._stall_watch.clear()
        self.logger.warning('[ACT] step=%d pos=%s STALL(cycle=%s) -> forcing %s | Q=[%s]',
                             step, pos, cycling_tiles, choice, _q_str())
        return choice

    # --- Greedy: argmax over valid actions ---
    q_legal = np.where(valid, q_vals, -np.inf)
    best_idx = np.flatnonzero(q_legal == np.max(q_legal))
    choice = ACTIONS[int(np.random.choice(best_idx))]
    self.logger.info('[ACT] step=%d pos=%s GREEDY -> %s | Q=[%s]',
                     step, pos, choice, _q_str())
    return choice


# ---------------------------------------------------------------------------
# Game-state helpers  (identical to ql_agent — shared logic)
# ---------------------------------------------------------------------------

def valid_action_mask(game_state):
    """Boolean mask over ACTIONS for currently legal moves."""
    field, bombs, others = game_state['field'], game_state['bombs'], game_state['others']
    _, _, bomb_possible, (x, y) = game_state['self']
    mask = [is_free(field, bombs, others, x + dx, y + dy) for _, dx, dy in DIRECTIONS]
    mask += [True, bomb_possible]
    return np.asarray(mask, dtype=bool)


def is_free(field, bombs, others, x, y):
    w, h = field.shape
    if not (0 <= x < w and 0 <= y < h) or field[x, y] != 0:
        return False
    if any((bx, by) == (x, y) for (bx, by), _ in bombs):
        return False
    return not any((ox, oy) == (x, y) for _, _, _, (ox, oy) in others)


def bfs(field, bombs, others, start, danger=None):
    """
    BFS over walkable tiles, optionally treating danger tiles as impassable.
    Returns (dist, first_step) dicts.
    """
    danger = danger or set()
    w, h = field.shape
    dist, first = {start: 0}, {start: None}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for name, dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            p = (nx, ny)
            if p in dist or p in danger:
                continue
            if not is_free(field, bombs, others, nx, ny):
                continue
            dist[p] = dist[(x, y)] + 1
            first[p] = first[(x, y)] or name
            queue.append(p)
    return dist, first


def compute_danger_tiles(field, bombs, explosion_map):
    """All tiles in an active explosion or any live bomb's future blast."""
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


def bomb_blast(field, pos):
    """All tiles hit by a bomb at pos (range 3, stopped by walls/crates)."""
    w, h = field.shape
    blast = {pos}
    x0, y0 = pos
    for _, dx, dy in DIRECTIONS:
        for step in range(1, 4):
            x, y = x0 + dx * step, y0 + dy * step
            if not (0 <= x < w and 0 <= y < h) or field[x, y] == -1:
                break
            blast.add((x, y))
            if field[x, y] == 1:
                break
    return blast


def tile_danger_at_time(field, bombs, explosion_map, pos, t):
    """True if pos is dangerous to occupy at time t."""
    if explosion_map[pos] > 0:
        return True
    for bomb_pos, timer in bombs:
        if t >= timer and pos in bomb_blast(field, bomb_pos):
            return True
    return False


def tile_permanently_safe(field, bombs, explosion_map, pos, from_t):
    """True if pos is safe at from_t and every future step until all bombs explode."""
    if not bombs and explosion_map[pos] <= 0:
        return True
    max_t = max((timer + 1 for _, timer in bombs), default=0)
    for t in range(from_t, max_t + 1):
        if tile_danger_at_time(field, bombs, explosion_map, pos, t):
            return False
    return True


def timed_escape_action(field, bombs, others, explosion_map, start, horizon=5):
    """
    Space-time BFS: find a first-step direction to a permanently safe tile.
    Returns None when already safe.
    """
    if not bombs and explosion_map[start] <= 0:
        return None

    bomb_tiles = {bp for bp, _ in bombs}
    max_timer = max((t for _, t in bombs), default=0)
    horizon = max(horizon, max_timer + 2)

    queue = deque([(start, 0, None)])
    visited = {(start, 0)}

    while queue:
        pos, t, first = queue.popleft()
        if t > 0 and tile_permanently_safe(field, bombs, explosion_map, pos, t):
            return first
        if t >= horizon:
            continue
        for name, dx, dy in DIRECTIONS:
            nxt = (pos[0] + dx, pos[1] + dy)
            w, h = field.shape
            if not (0 <= nxt[0] < w and 0 <= nxt[1] < h):
                continue
            if field[nxt[0], nxt[1]] != 0:
                continue
            if nxt in bomb_tiles:
                continue
            if any((ox, oy) == nxt for _, _, _, (ox, oy) in others):
                continue
            nt = t + 1
            if tile_danger_at_time(field, bombs, explosion_map, nxt, nt):
                continue
            state = (nxt, nt)
            if state in visited:
                continue
            visited.add(state)
            queue.append((nxt, nt, first or name))

    return None


def is_safe_to_bomb(field, bombs, others, pos, explosion_map=None):
    """True if dropping a bomb at pos leaves a timed escape route."""
    if explosion_map is None:
        explosion_map = np.zeros_like(field, dtype=np.float32)
    hypothetical = list(bombs) + [(pos, 3)]
    horizon = max(4, max((t for _, t in hypothetical), default=3) + 1)
    return timed_escape_action(
        field, hypothetical, others, explosion_map, pos, horizon=horizon,
    ) is not None


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


def direction_onehot(direction):
    return [int(direction == name) for name, _, _ in DIRECTIONS]


def nearest_target_direction(dist, first_step, targets):
    reachable = [t for t in targets if t in dist]
    return first_step[min(reachable, key=dist.get)] if reachable else None


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


def nearest_coin_distance(game_state):
    if not game_state['coins']:
        return None
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
    if not targets:
        return None
    danger = compute_danger_tiles(field, bombs, game_state['explosion_map'])
    dist, _ = bfs(field, bombs, others, pos, danger)
    vals = [dist[t] for t in targets if t in dist]
    return min(vals) if vals else None


def nearest_opponent_info(field, bombs, others, danger, pos):
    """
    Returns (direction_onehot, hits_opponent_flag, normalised_distance).
    """
    if not others:
        return [0, 0, 0, 0], 0, 1.0
    dist, first = bfs(field, bombs, others, pos, danger)
    opp_positions = [p for _, _, _, p in others]
    reachable = [p for p in opp_positions if p in dist]
    if reachable:
        nearest = min(reachable, key=dist.get)
        opp_dir = direction_onehot(first[nearest])
        opp_dist = min(dist[nearest] / 10.0, 1.0)
    else:
        opp_dir = [0, 0, 0, 0]
        opp_dist = 1.0
    hits_opp = int(bomb_would_hit_opponents(field, pos, others))
    return opp_dir, hits_opp, opp_dist


def state_to_features(game_state):
    if game_state is None:
        return None

    field = game_state['field']
    bombs = game_state['bombs']
    others = game_state['others']
    coins = game_state['coins']
    explosion_map = game_state['explosion_map']
    _, _, bomb_possible, pos = game_state['self']

    danger = compute_danger_tiles(field, bombs, explosion_map)
    dist, first = bfs(field, bombs, others, pos, danger)

    # [1..4] Coin direction
    coin_dir = direction_onehot(nearest_target_direction(dist, first, coins))

    # [5..8] Free neighbours
    x, y = pos
    free = [
        int(is_free(field, bombs, others, x + dx, y + dy))
        for _, dx, dy in DIRECTIONS
    ]

    # [9] In danger
    in_danger = int(pos in danger)

    # [10..13] Timed escape direction
    max_timer = max((t for _, t in bombs), default=0)
    timed_escape = timed_escape_action(
        field, bombs, others, explosion_map, pos,
        horizon=max(5, max_timer + 2),
    )
    escape = direction_onehot(timed_escape)

    # [14] Bomb available
    bp = int(bomb_possible)

    # [15..18] Crate direction
    crates = crate_adjacent_tiles(field)
    crate_dir = direction_onehot(nearest_target_direction(dist, first, crates))

    # [19] Bomb hits a crate
    hits_crate = int(bomb_would_hit_crate(field, pos))

    # [20] Safe to bomb right now
    safe_bomb = int(
        bomb_possible and is_safe_to_bomb(field, bombs, others, pos, explosion_map)
    )

    # [21..24] Nearest opponent direction, [25] bomb hits opponent, [26] normalised dist
    opp_dir, hits_opp, opp_dist = nearest_opponent_info(field, bombs, others, danger, pos)

    features = np.array(
        [1.0]
        + coin_dir     # 4
        + free         # 4
        + [in_danger]  # 1
        + escape       # 4
        + [bp]         # 1
        + crate_dir    # 4
        + [hits_crate, safe_bomb]  # 2
        + opp_dir      # 4
        + [hits_opp, opp_dist],    # 2
        dtype=np.float64,
    )
    assert len(features) == FEATURE_DIM, f'Feature length {len(features)} != {FEATURE_DIM}'
    return features