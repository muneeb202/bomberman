"""Runtime callbacks for the DDQN Bomberman agent.

This first slice intentionally contains no learning logic.  It establishes the
framework contract and keeps the action mapping in one place for later DDQN
components.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Optional

import numpy as np
import torch

from .model import DuelingQNetwork, QNetwork


# Keep this order stable: network output index i always represents ACTIONS[i].
ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}

# A local view supplies spatial context while keeping a CPU MLP small. Four
# global nearest-coin values supplement the local view when no coin is nearby.
# The standard representation has 509 values. Optional research inputs append
# values while keeping legacy 509-feature checkpoints usable.
VIEW_RADIUS = 4
VIEW_SIZE = 2 * VIEW_RADIUS + 1
USE_PROSPECTIVE_BOMB_SAFETY = os.environ.get("DDQN_USE_PROSPECTIVE_BOMB_SAFETY", "0") == "1"
USE_ACTION_ESCAPE_FEATURES = os.environ.get("DDQN_USE_ACTION_ESCAPE_FEATURES", "0") == "1"
FEATURE_DIM = (
    6 * VIEW_SIZE * VIEW_SIZE
    + 3
    + 4
    + 4
    + len(ACTIONS)
    + len(ACTIONS)
    + int(USE_PROSPECTIVE_BOMB_SAFETY)
    + len(ACTIONS) * int(USE_ACTION_ESCAPE_FEATURES)
)
# The packaged default is a compatible, selected checkpoint. Earlier
# experiment checkpoints remain separately preserved for reproducibility.
MODEL_FILE = os.environ.get("DDQN_MODEL_FILE", "ddqn_model.pt")
ARCHITECTURE = os.environ.get("DDQN_ARCHITECTURE", "dueling")
if ARCHITECTURE not in {"standard", "dueling"}:
    raise ValueError("DDQN_ARCHITECTURE must be 'standard' or 'dueling'")

_ACTION_DELTAS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
}


def setup(self) -> None:
    """Initialise persistent inference state once per agent instance."""
    self.device = torch.device("cpu")
    network_class = DuelingQNetwork if ARCHITECTURE == "dueling" else QNetwork
    self.q_network = network_class(FEATURE_DIM, len(ACTIONS)).to(self.device)
    self.rng = np.random.default_rng()
    if os.path.isfile(MODEL_FILE):
        checkpoint = torch.load(MODEL_FILE, map_location=self.device)
        checkpoint_architecture = checkpoint.get("architecture", "standard")
        if (
            checkpoint.get("feature_dim") != FEATURE_DIM
            or checkpoint.get("actions") != ACTIONS
            or checkpoint_architecture != ARCHITECTURE
        ):
            raise ValueError("Checkpoint does not match this ddqn_agent feature/action configuration")
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.logger.info("Loaded DDQN checkpoint from %s.", MODEL_FILE)
    else:
        self.logger.info("No checkpoint found; initialising DDQN network from scratch.")
    self.q_network.eval()
    self.action_count = 0
    self.diagnostic_round = None
    self.diagnostic_positions = deque(maxlen=4)
    self.diagnostic_wait_streak = 0
    self.diagnostic_max_wait_streak = 0
    self.diagnostic_reversals = 0
    self.diagnostic_four_step_revisits = 0


def _start_diagnostic_round(self, round_id: int) -> None:
    """Log compact, evaluation-safe stalling diagnostics for the prior round."""
    if self.diagnostic_round is not None:
        self.logger.info(
            "DDQN diagnostics round=%s max_wait_streak=%s reversals=%s four_step_revisits=%s",
            self.diagnostic_round,
            self.diagnostic_max_wait_streak,
            self.diagnostic_reversals,
            self.diagnostic_four_step_revisits,
        )
    self.diagnostic_round = round_id
    self.diagnostic_positions.clear()
    self.diagnostic_wait_streak = 0
    self.diagnostic_max_wait_streak = 0
    self.diagnostic_reversals = 0
    self.diagnostic_four_step_revisits = 0


def act(self, game_state: dict) -> str:
    """Return a valid framework action.

    WAIT is deliberately used during Phase 1.  It is always accepted by the
    environment and lets us verify both evaluation and training callbacks
    without mixing in navigation or learning behaviour.
    """
    if game_state["round"] != self.diagnostic_round:
        _start_diagnostic_round(self, game_state["round"])
    position = game_state["self"][3]
    if len(self.diagnostic_positions) >= 2 and position == self.diagnostic_positions[-2]:
        self.diagnostic_reversals += 1
    if len(self.diagnostic_positions) == 4 and position == self.diagnostic_positions[0]:
        self.diagnostic_four_step_revisits += 1
    self.diagnostic_positions.append(position)

    features = state_to_features(game_state)
    self.last_action_features = features
    self.last_feature_shape = features.shape
    self.action_count += 1
    valid_actions = np.flatnonzero(action_mask(game_state))
    epsilon = getattr(self, "epsilon", 0.0)
    if self.train and self.rng.random() < epsilon:
        action = ACTIONS[int(self.rng.choice(valid_actions))]
        if action == "WAIT":
            self.diagnostic_wait_streak += 1
        else:
            self.diagnostic_wait_streak = 0
        self.diagnostic_max_wait_streak = max(self.diagnostic_max_wait_streak, self.diagnostic_wait_streak)
        return action

    with torch.no_grad():
        q_values = self.q_network(torch.from_numpy(features).unsqueeze(0)).squeeze(0).numpy()
    q_values[action_mask(game_state) == 0] = -np.inf
    action = ACTIONS[int(np.argmax(q_values))]
    if action == "WAIT":
        self.diagnostic_wait_streak += 1
    else:
        self.diagnostic_wait_streak = 0
    self.diagnostic_max_wait_streak = max(self.diagnostic_max_wait_streak, self.diagnostic_wait_streak)
    return action


def action_mask(game_state: dict) -> np.ndarray:
    """Return actions accepted by the environment before future danger is considered.

    The mask mirrors ``BombeRLeWorld.tile_is_free`` for movements.  It is an
    action-legality signal, not a rule-based safety policy: stepping into a
    future bomb blast remains available for the learned Q-function to reject.
    """
    field = game_state["field"]
    _, _, bombs_left, (x, y) = game_state["self"]
    occupied = {position for position, _ in game_state["bombs"]}
    occupied.update(other[3] for other in game_state["others"])

    mask = np.zeros(len(ACTIONS), dtype=np.float32)
    for action, (dx, dy) in _ACTION_DELTAS.items():
        nx, ny = x + dx, y + dy
        if field[nx, ny] == 0 and (nx, ny) not in occupied:
            mask[ACTION_TO_INDEX[action]] = 1.0
    mask[ACTION_TO_INDEX["WAIT"]] = 1.0
    mask[ACTION_TO_INDEX["BOMB"]] = float(bool(bombs_left))
    return mask


def _local_view(array: np.ndarray, x: int, y: int, fill_value: float) -> np.ndarray:
    """Extract a padded agent-centred square without changing x/y semantics."""
    view = np.full((VIEW_SIZE, VIEW_SIZE), fill_value, dtype=np.float32)
    for local_x, board_x in enumerate(range(x - VIEW_RADIUS, x + VIEW_RADIUS + 1)):
        for local_y, board_y in enumerate(range(y - VIEW_RADIUS, y + VIEW_RADIUS + 1)):
            if 0 <= board_x < array.shape[0] and 0 <= board_y < array.shape[1]:
                view[local_x, local_y] = array[board_x, board_y]
    return view


def _bomb_danger_map(game_state: dict) -> np.ndarray:
    """Map bomb blast squares to urgency, matching the framework's blast rules."""
    field = game_state["field"]
    danger = np.zeros(field.shape, dtype=np.float32)
    width, height = field.shape
    for (bomb_x, bomb_y), timer in game_state["bombs"]:
        urgency = 1.0 - min(max(float(timer), 0.0), 4.0) / 4.0
        danger[bomb_x, bomb_y] = max(danger[bomb_x, bomb_y], urgency)
        for dx, dy in _ACTION_DELTAS.values():
            for distance in range(1, 4):
                nx, ny = bomb_x + dx * distance, bomb_y + dy * distance
                if not (0 <= nx < width and 0 <= ny < height) or field[nx, ny] == -1:
                    break
                danger[nx, ny] = max(danger[nx, ny], urgency)
    return danger


def action_immediate_safety(game_state: dict) -> np.ndarray:
    """Encode immediate safety of each action without selecting an action.

    A value of one means the legal action does not leave the agent in an active
    explosion or in a bomb blast scheduled for the next environment step.
    """
    field = game_state["field"]
    _, _, _, (x, y) = game_state["self"]
    dangerous = set(zip(*np.where(game_state["explosion_map"] > 0)))
    for position, timer in game_state["bombs"]:
        if timer <= 1:
            bx, by = position
            dangerous.add(position)
            for dx, dy in _ACTION_DELTAS.values():
                for distance in range(1, 4):
                    nx, ny = bx + dx * distance, by + dy * distance
                    if field[nx, ny] == -1:
                        break
                    dangerous.add((nx, ny))
    legal = action_mask(game_state)
    safe = np.zeros(len(ACTIONS), dtype=np.float32)
    for action, (dx, dy) in _ACTION_DELTAS.items():
        target = (x + dx, y + dy)
        safe[ACTION_TO_INDEX[action]] = float(legal[ACTION_TO_INDEX[action]] and target not in dangerous)
    safe[ACTION_TO_INDEX["WAIT"]] = float((x, y) not in dangerous)
    safe[ACTION_TO_INDEX["BOMB"]] = float(legal[ACTION_TO_INDEX["BOMB"]] and (x, y) not in dangerous)
    return safe


def _blast_tiles(field: np.ndarray, position: tuple[int, int]) -> set[tuple[int, int]]:
    """Return tiles reached by a bomb blast under the agent's wall rule."""
    x, y = position
    tiles = {position}
    for dx, dy in _ACTION_DELTAS.values():
        for distance in range(1, 4):
            nx, ny = x + dx * distance, y + dy * distance
            if field[nx, ny] == -1:
                break
            tiles.add((nx, ny))
    return tiles


def action_escape_route_safety(game_state: dict, horizon: int = 4) -> np.ndarray:
    """Return whether each legal action has a safe route through bomb fuses.

    This is an input feature, never an action mask.  For every candidate action
    it performs a small time-expanded search over the next four game steps. A
    candidate is safe when at least one legal sequence (including waiting) can
    avoid active explosions and every blast that occurs in that horizon. For a
    BOMB candidate, the newly placed bomb is included with its four-step fuse.
    """
    field = game_state["field"]
    _, _, _, start = game_state["self"]
    legal = action_mask(game_state)
    current_explosions = set(zip(*np.where(game_state["explosion_map"] > 0)))
    bombs = list(game_state["bombs"])

    def route_exists(action: str) -> bool:
        index = ACTION_TO_INDEX[action]
        if not legal[index]:
            return False
        if action in _ACTION_DELTAS:
            dx, dy = _ACTION_DELTAS[action]
            first_position = (start[0] + dx, start[1] + dy)
        else:
            first_position = start

        timed_bombs = list(bombs)
        if action == "BOMB":
            timed_bombs.append((start, horizon))
        danger_by_step = [set() for _ in range(horizon + 1)]
        blocked_by_step = [set() for _ in range(horizon + 1)]
        # An active explosion is unsafe on the first resulting state.
        danger_by_step[1].update(current_explosions)
        for bomb_position, timer in timed_bombs:
            explosion_step = max(1, min(int(timer), horizon))
            danger_by_step[explosion_step].update(_blast_tiles(field, bomb_position))
            for step in range(1, explosion_step + 1):
                blocked_by_step[step].add(bomb_position)

        if first_position in danger_by_step[1]:
            return False
        frontier = {first_position}
        for step in range(2, horizon + 1):
            next_frontier = set()
            for x, y in frontier:
                for dx, dy in (*_ACTION_DELTAS.values(), (0, 0)):
                    candidate = (x + dx, y + dy)
                    if (
                        field[candidate[0], candidate[1]] == 0
                        and candidate not in blocked_by_step[step]
                        and candidate not in danger_by_step[step]
                    ):
                        next_frontier.add(candidate)
            frontier = next_frontier
            if not frontier:
                return False
        return True

    return np.array([float(route_exists(action)) for action in ACTIONS], dtype=np.float32)


def prospective_bomb_escape_signal(game_state: dict) -> float:
    """Return whether a bomb can clear an adjacent crate and be escaped.

    This is an input to the learned policy, not an action override. It models
    the newly placed bomb as occupying the current tile and checks whether a
    free tile outside its blast is reachable within its four-tick fuse.
    Existing-bomb and explosion channels remain separate inputs for the MLP.
    """
    field = game_state["field"]
    _, _, bombs_left, start = game_state["self"]
    if not bombs_left or game_state["explosion_map"][start] > 0:
        return 0.0
    if not any(field[start[0] + dx, start[1] + dy] == 1 for dx, dy in _ACTION_DELTAS.values()):
        return 0.0

    blast_tiles = {start}
    for dx, dy in _ACTION_DELTAS.values():
        for distance in range(1, 4):
            nx, ny = start[0] + dx * distance, start[1] + dy * distance
            if field[nx, ny] == -1:
                break
            blast_tiles.add((nx, ny))

    occupied_bombs = {position for position, _ in game_state["bombs"]}
    frontier = {start}
    for _ in range(4):
        next_frontier = set()
        for x, y in frontier:
            for dx, dy in _ACTION_DELTAS.values():
                nx, ny = x + dx, y + dy
                if field[nx, ny] == 0 and (nx, ny) != start and (nx, ny) not in occupied_bombs:
                    next_frontier.add((nx, ny))
        frontier = next_frontier
        if not frontier:
            return 0.0
    return float(any(position not in blast_tiles for position in frontier))


def nearest_reachable_coin_info(game_state: dict) -> tuple[float, float, float, float]:
    """Return direction and shortest path distance to the nearest free coin.

    The values are normalized horizontal offset, normalized vertical offset,
    normalized shortest-path distance, and a reachable-coin indicator.  The
    feature provides global navigation context but does not prescribe an action.
    """
    field = game_state["field"]
    _, _, _, start = game_state["self"]
    targets = set(game_state["coins"])
    if not targets:
        return 0.0, 0.0, 0.0, 0.0

    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        (x, y), distance = queue.popleft()
        if (x, y) in targets:
            max_x = field.shape[0] - 1
            max_y = field.shape[1] - 1
            max_distance = max_x + max_y
            return (
                (x - start[0]) / max_x,
                (y - start[1]) / max_y,
                min(distance, max_distance) / max_distance,
                1.0,
            )
        for dx, dy in _ACTION_DELTAS.values():
            nx, ny = x + dx, y + dy
            if (nx, ny) not in visited and field[nx, ny] == 0:
                visited.add((nx, ny))
                queue.append(((nx, ny), distance + 1))
    return 0.0, 0.0, 0.0, 0.0


def nearest_reachable_crate_info(game_state: dict) -> tuple[float, float, float, float]:
    field = game_state["field"]
    _, _, _, start = game_state["self"]
    queue, visited = deque([(start, 0)]), {start}
    while queue:
        (x, y), distance = queue.popleft()
        if any(field[x + dx, y + dy] == 1 for dx, dy in _ACTION_DELTAS.values()):
            return ((x-start[0])/16, (y-start[1])/16, min(distance, 32)/32, 1.0)
        for dx, dy in _ACTION_DELTAS.values():
            p = (x + dx, y + dy)
            if p not in visited and field[p[0], p[1]] == 0:
                visited.add(p); queue.append((p, distance + 1))
    return 0.0, 0.0, 0.0, 0.0


def state_to_features(game_state: Optional[dict]) -> Optional[np.ndarray]:
    """Encode a game state as a fixed 509-value vector plus optional inputs.

    Six agent-centred local channels represent terrain, coins, opponents, bomb
    timers, bomb-blast urgency, and active explosions. Global coordinates,
    bomb availability, global nearest-coin navigation, and the legality mask
    complement the local context.
    """
    if game_state is None:
        return None

    field = game_state["field"]
    _, _, bombs_left, (x, y) = game_state["self"]
    coins = np.zeros(field.shape, dtype=np.float32)
    opponents = np.zeros(field.shape, dtype=np.float32)
    bomb_timers = np.zeros(field.shape, dtype=np.float32)
    for coin_x, coin_y in game_state["coins"]:
        coins[coin_x, coin_y] = 1.0
    for _, _, _, (other_x, other_y) in game_state["others"]:
        opponents[other_x, other_y] = 1.0
    for (bomb_x, bomb_y), timer in game_state["bombs"]:
        bomb_timers[bomb_x, bomb_y] = max(bomb_timers[bomb_x, bomb_y], float(timer) / 4.0)

    channels = (
        _local_view(field.astype(np.float32), x, y, -1.0),
        _local_view(coins, x, y, 0.0),
        _local_view(opponents, x, y, 0.0),
        _local_view(bomb_timers, x, y, 0.0),
        _local_view(_bomb_danger_map(game_state), x, y, 0.0),
        _local_view(game_state["explosion_map"].astype(np.float32) / 2.0, x, y, 0.0),
    )
    extras = np.array(
        [
            x / (field.shape[0] - 1),
            y / (field.shape[1] - 1),
            float(bool(bombs_left)),
            *nearest_reachable_coin_info(game_state),
            *nearest_reachable_crate_info(game_state),
        ],
        dtype=np.float32,
    )
    feature_parts = [
        *(channel.ravel() for channel in channels),
        extras,
        action_mask(game_state),
        action_immediate_safety(game_state),
    ]
    if USE_PROSPECTIVE_BOMB_SAFETY:
        feature_parts.append(np.array([prospective_bomb_escape_signal(game_state)], dtype=np.float32))
    if USE_ACTION_ESCAPE_FEATURES:
        feature_parts.append(action_escape_route_safety(game_state))
    features = np.concatenate(feature_parts)
    assert features.shape == (FEATURE_DIM,)
    return features.astype(np.float32, copy=False)
