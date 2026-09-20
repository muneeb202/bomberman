import os
import pickle
import random
from collections import deque
import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DIRECTIONS = [('UP', 0, -1), ('RIGHT', 1, 0), ('DOWN', 0, 1), ('LEFT', -1, 0)]

# Feature vector layout (total = 33). Scope: Task 1+2 (coins + crates, no
# opponents yet) -- opponent features can be added later for Task 3+4.
#   [0]       bias_danger  -- on only when in_danger, i.e. index [9] below is 1
#   [1]       bias_safe    -- on only when NOT in_danger
#             These two replace a single always-on bias term. A single
#             shared bias can't represent "WAIT is good conditional on
#             danger" -- a TD update during a (correctly) high-value WAIT
#             in a danger state touches every active feature, including a
#             constant bias, which then leaks a WAIT baseline into every
#             other state regardless of context. Splitting the bias by
#             danger/safe context lets each learn its own baseline.
#   [2..5]    coin direction one-hot  (UP/RIGHT/DOWN/LEFT), 0s if no reachable coin
#   [6..9]    free neighbours         (UP/RIGHT/DOWN/LEFT)
#   [10]      in_danger flag
#   [11..14]  timed escape direction  (UP/RIGHT/DOWN/LEFT), 0s if already safe
#   [15]      bomb_possible
#   [16..19]  nearest crate-adjacent direction (falls back to a BFS that
#             treats crates as passable when no free-tile-only path exists,
#             so this signal never goes fully blank)
#   [20]      bomb_would_hit_crate
#   [21]      safe_to_bomb
#   [22]      no_objective -- 1 when both coin_dir and crate_dir are all-zero,
#             i.e. genuinely nothing to chase right now. Without this, "no
#             objective" is only inferable indirectly (four other features
#             all reading zero), which is a weaker signal for the linear
#             model to key off of than a single dedicated indicator.
#   [23..26]  last move direction one-hot (UP/RIGHT/DOWN/LEFT), 0s if the
#             last action was WAIT/BOMB or this is the first step of the
#             round. This gives the Q-function enough information to learn,
#             per action, whether immediately reversing the previous move
#             tends to pay off -- without it, two tiles whose BFS-based
#             features are identical can form a fully self-consistent
#             back-and-forth fixed point that no amount of training fixes.
#   [27..30]  nearest opponent direction one-hot, 0s if no opponent
#   [31]      bomb_would_hit_opponent
#   [32]      nearest opponent distance (normalised by 10, clamped to 1.0)
FEATURE_DIM = 33


def setup(self):
    shape = (2, len(ACTIONS), FEATURE_DIM)
    if os.path.isfile('my-saved-model.pt'):
        self.logger.info('Loading model from saved state (resuming).')
        with open('my-saved-model.pt', 'rb') as f:
            self.model = pickle.load(f)
        if self.model.shape != shape:
            self.logger.warning(
                'Incompatible saved model shape %s vs %s; starting fresh.',
                self.model.shape, shape,
            )
            self.model = np.zeros(shape, dtype=np.float64)
    else:
        self.logger.info('No saved model found; setting up from scratch (shape=%s).', shape)
        self.model = np.zeros(shape, dtype=np.float64)

    self.last_action = None
    self._last_features = None


INFERENCE_EPSILON = 0.02  # small constant exploration floor kept even when
# self.train is False. With a linear Q-function over hand-crafted features,
# two dissimilar states can end up with near-tied Q-values for different
# actions purely from generalisation crosstalk (e.g. WAIT is correctly very
# valuable while riding out an explosion, and that weight can bleed into an
# unrelated idle state where it's actually a hair below the better option).
# A razor-thin, incorrect tie at a state that maps back to itself (WAIT ->
# same state) is a permanent fixed point under pure greedy play -- nothing
# will ever perturb it again. This doesn't override or replace any learned
# decision; the model's own ranking is used the other 98% of the time.


def act(self, game_state):
    """
    Epsilon-greedy over the Double-Q function. Only physically invalid
    actions (walls, occupied tiles, no bomb available) are masked out.
    All behaviour -- navigation, bombing, escape, not reversing into a
    loop -- is learned from the reward signal and the state features,
    not hard-coded here.
    """
    epsilon = getattr(self, 'epsilon', 0.05) if self.train else INFERENCE_EPSILON
    valid = valid_action_mask(game_state)
    step = game_state['step']
    pos = game_state['self'][3]

    if not hasattr(self, 'last_action'):
        self.last_action = None

    features = state_to_features(game_state, self.last_action)
    self._last_features = features

    q_avg = (self.model[0] + self.model[1]) @ features / 2.0
    q_masked = np.where(valid, q_avg, np.nan)

    def _q_str():
        parts = []
        for a, v in zip(ACTIONS, q_masked):
            parts.append(f'{a}:{"off" if np.isnan(v) else f"{v:+.3f}"}')
        return ' | '.join(parts)

    if random.random() < epsilon:
        legal = [a for a, ok in zip(ACTIONS, valid) if ok]
        choice = random.choice(legal)
        self.logger.info('[ACT] step=%d pos=%s eps=%.3f EXPLORE -> %s | Q=[%s]',
                          step, pos, epsilon, choice, _q_str())
        self.last_action = choice
        return choice

    q_legal = np.where(valid, q_avg, -np.inf)
    best_idx = np.flatnonzero(q_legal == np.max(q_legal))
    choice = ACTIONS[int(np.random.choice(best_idx))]
    self.logger.info('[ACT] step=%d pos=%s GREEDY -> %s | Q=[%s]', step, pos, choice, _q_str())
    self.last_action = choice
    return choice


# ---------------------------------------------------------------------------
# Game-state helpers
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
    BFS over currently walkable tiles (walls/crates block, and so do
    bomb tiles and other agents), optionally treating `danger` tiles
    (an active or imminent blast) as impassable too. Used for coin
    direction/distance -- without danger avoidance, a "shortest path"
    can route straight through a live bomb.
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


def bfs_through_crates(field, bombs, others, start, danger=None):
    """
    Same as bfs(), but treats crate tiles as passable, representing
    "reachable once bombed through". The strict bfs() blocks crates
    exactly like walls, so a crate-targeting distance/direction based on
    it alone goes fully silent the moment every remaining crate is sealed
    behind another crate instead of sitting next to already-cleared
    floor -- which is the normal state of the map after the first ring
    of crates is destroyed. Used only as a fallback when the strict
    search finds nothing, so a directional signal is always available.
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
            if not (0 <= nx < w and 0 <= ny < h) or field[nx, ny] == -1:
                continue
            if any((bx, by) == p for (bx, by), _ in bombs):
                continue
            if any((ox, oy) == p for _, _, _, (ox, oy) in others):
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
    """
    True if pos is dangerous to occupy at time t. A tile in a bomb's
    blast radius is deadly exactly when the bomb fires (t == timer) and
    briefly at t == timer+1 for the lingering explosion.
    """
    if explosion_map[pos] > 0:
        return True
    for bomb_pos, timer in bombs:
        if t >= timer and pos in bomb_blast(field, bomb_pos):
            return True
    return False


def tile_permanently_safe(field, bombs, explosion_map, pos, from_t):
    """
    True if pos is safe at time from_t AND at every future time step
    until all current bombs have fully exploded (timer + 1 for lingering).
    """
    if not bombs and explosion_map[pos] <= 0:
        return True
    max_t = max((timer + 1 for _, timer in bombs), default=0)
    for t in range(from_t, max_t + 1):
        if tile_danger_at_time(field, bombs, explosion_map, pos, t):
            return False
    return True


def timed_escape_action(field, bombs, others, explosion_map, start, horizon=5):
    """
    Space-time BFS: find a first-step direction that leads the agent to a
    tile that is permanently safe (not hit by any bomb at any future
    time). Returns None when already safe (no bombs, no active explosion).
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
    """
    True if dropping a bomb at pos leaves a timed escape route.

    Timer convention: the agent places the bomb; the environment
    decrements it in the same step so the NEXT game_state shows timer=3.
    We model the new bomb with timer=3 to match what the escape BFS will
    face when it runs on the very next step.
    """
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
    if vals:
        return min(vals)
    dist, _ = bfs_through_crates(field, bombs, others, pos, danger)
    vals = [dist[t] for t in targets if t in dist]
    return min(vals) if vals else None


def nearest_opponent_distance(game_state):
    others = game_state['others']
    if not others:
        return None
    field, bombs = game_state['field'], game_state['bombs']
    pos = game_state['self'][3]
    danger = compute_danger_tiles(field, bombs, game_state['explosion_map'])
    dist, _ = bfs(field, bombs, others, pos, danger)
    vals = [dist[p] for _, _, _, p in others if p in dist]
    return min(vals) if vals else None


def state_to_features(game_state, last_action=None):
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

    # [5..8] Free neighbours (physical walkability this step)
    x, y = pos
    free = [
        int(is_free(field, bombs, others, x + dx, y + dy))
        for _, dx, dy in DIRECTIONS
    ]

    # [9] In danger
    in_danger = int(pos in danger)

    # [10..13] Timed escape direction (zeros when already safe)
    max_timer = max((t for _, t in bombs), default=0)
    timed_escape = timed_escape_action(
        field, bombs, others, explosion_map, pos,
        horizon=max(5, max_timer + 2),
    )
    escape = direction_onehot(timed_escape)

    # [14] Bomb available
    bp = int(bomb_possible)

    # [15..18] Crate direction (with permeable-BFS fallback so it's never
    # silent just because the direct path requires bombing through first)
    crates = crate_adjacent_tiles(field)
    crate_target = nearest_target_direction(dist, first, crates)
    if crate_target is None and crates:
        dist_via_crates, first_via_crates = bfs_through_crates(field, bombs, others, pos, danger)
        crate_target = nearest_target_direction(dist_via_crates, first_via_crates, crates)
    crate_dir = direction_onehot(crate_target)

    # [19] Bomb hits a crate
    hits_crate = int(bomb_would_hit_crate(field, pos))

    # [20] Safe to bomb right now
    safe_bomb = int(
        bomb_possible and is_safe_to_bomb(field, bombs, others, pos, explosion_map)
    )

    # [22] No objective (nothing to chase right now)
    no_objective = int(sum(coin_dir) == 0 and sum(crate_dir) == 0)

    # [23..26] Last move direction (0s if last action wasn't a move)
    last_move_dir = direction_onehot(last_action if last_action in ('UP', 'RIGHT', 'DOWN', 'LEFT') else None)

    # [27..30] Nearest opponent direction, [31] bomb hits opponent, [32] normalised dist
    opp_dir, hits_opp, opp_dist = nearest_opponent_info(field, bombs, others, danger, pos)

    features = np.array(
        [float(in_danger), float(1 - in_danger)]  # bias_danger, bias_safe
        + coin_dir     # 4
        + free         # 4
        + [in_danger]  # 1
        + escape       # 4
        + [bp]         # 1
        + crate_dir    # 4
        + [hits_crate, safe_bomb]  # 2
        + [no_objective]  # 1
        + last_move_dir  # 4
        + opp_dir      # 4
        + [hits_opp, opp_dist],  # 2
        dtype=np.float64,
    )
    assert len(features) == FEATURE_DIM, f'Feature length {len(features)} != {FEATURE_DIM}'
    return features
