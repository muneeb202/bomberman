"""CPU-only Double DQN training callbacks."""

from __future__ import annotations

import copy
import os
from collections import deque
from typing import List

import torch
from torch import nn

import events as e
from .callbacks import ARCHITECTURE, ACTIONS, ACTION_TO_INDEX, FEATURE_DIM, MODEL_FILE, action_mask, state_to_features
from .model import ReplayBuffer


GAMMA = 0.99
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
REPLAY_CAPACITY = 25_000
LEARNING_STARTS = 1_000
TRAIN_EVERY = 4
TARGET_SYNC_EVERY = 1_000
EPSILON_START = float(os.environ.get("DDQN_EPSILON_START", "1.0"))
EPSILON_END = float(os.environ.get("DDQN_EPSILON_END", "0.05"))
SELF_DEATH_PENALTY = float(os.environ.get("DDQN_SELF_DEATH_PENALTY", "-12.0"))
OPPONENT_DEATH_PENALTY = float(os.environ.get("DDQN_OPPONENT_DEATH_PENALTY", "-12.0"))
# E11 uses a shorter schedule because loot-crate rounds initially provide far
# fewer transitions than coin-heaven; the original 100k schedule left epsilon
# near 0.91 after 1,000 Task-2 rounds.
EPSILON_DECAY_STEPS = 30_000

EVENT_REWARDS = {
    e.COIN_COLLECTED: 5.0,
    e.CRATE_DESTROYED: 1.0,
    e.KILLED_OPPONENT: 10.0,
    e.KILLED_SELF: SELF_DEATH_PENALTY,
    e.GOT_KILLED: OPPONENT_DEATH_PENALTY,
    e.INVALID_ACTION: -1.0,
    e.WAITED: -0.02,
    e.SURVIVED_ROUND: 2.0,
}

# Retained to make the documented E04 reward ablation reproducible. The E01
# baseline remains active after E04 reduced the matched coin-collection score.
USE_COIN_PROGRESS_SHAPING = False
COIN_PROGRESS_REWARD = 0.10
_NEIGHBOUR_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))

# E10: encourage only bomb placements that can immediately clear a crate and
# leave enough free-tile routes to escape the blast.
USE_SAFE_CRATE_BOMB_SHAPING = True
SAFE_CRATE_BOMB_REWARD = float(os.environ.get("DDQN_SAFE_CRATE_BOMB_REWARD", "0.75"))
UNSAFE_OR_WASTED_BOMB_PENALTY = float(
    os.environ.get("DDQN_UNSAFE_OR_WASTED_BOMB_PENALTY", "-0.50")
)

# E12: discourage the simplest observed movement cycle, A -> B -> A. This is
# a training-only reward signal; inference remains entirely learned.
USE_IMMEDIATE_REVERSAL_PENALTY = True
IMMEDIATE_REVERSAL_PENALTY = -0.20

# E39: the E12 signal cannot identify longer unproductive cycles such as
# A -> B -> C -> B.  Penalise a return to any of the last four positions only
# when the transition did not produce an objective event.  The default is zero
# so historical experiments and ordinary runs retain their documented reward.
USE_SHORT_CYCLE_PENALTY = os.environ.get("DDQN_USE_SHORT_CYCLE_PENALTY", "0") == "1"
SHORT_CYCLE_PENALTY = float(os.environ.get("DDQN_SHORT_CYCLE_PENALTY", "-0.05"))
_OBJECTIVE_EVENTS = {e.COIN_COLLECTED, e.CRATE_DESTROYED, e.KILLED_OPPONENT}

# E28: a single WAIT can be tactically valid, but repeated waits were observed
# after a crate-clearing bomb. Penalise only the second and later consecutive
# waits so the learner receives a clear anti-stalling signal.
USE_CONSECUTIVE_WAIT_PENALTY = True
CONSECUTIVE_WAIT_PENALTY = float(os.environ.get("DDQN_CONSECUTIVE_WAIT_PENALTY", "-0.20"))
CONSECUTIVE_WAIT_THRESHOLD = int(os.environ.get("DDQN_CONSECUTIVE_WAIT_THRESHOLD", "2"))


def setup_training(self) -> None:
    """Create DDQN training state after inference setup has loaded the network."""
    self.target_network = copy.deepcopy(self.q_network).to(self.device)
    self.target_network.eval()
    self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
    self.loss_fn = nn.SmoothL1Loss()
    self.replay_buffer = ReplayBuffer(REPLAY_CAPACITY, FEATURE_DIM, len(ACTIONS))
    self.environment_steps = 0
    self.gradient_steps = 0
    self.completed_rounds = 0
    self.last_loss = None
    self.last_recorded_state_step = None
    # Four positions let E39 identify both direct reversals and longer local
    # movement cycles.  E12 itself still uses the same two-position test.
    self.recent_positions = deque(maxlen=4)
    self.consecutive_waits = 0
    self.epsilon = EPSILON_START
    self.logger.info("DDQN training ready: replay=%s, batch=%s, warmup=%s", REPLAY_CAPACITY, BATCH_SIZE, LEARNING_STARTS)


def nearest_reachable_coin_distance(game_state: dict) -> int | None:
    """Find the shortest free-tile path to a currently collectable coin."""
    if game_state is None or not game_state["coins"]:
        return None
    field = game_state["field"]
    _, _, _, start = game_state["self"]
    targets = set(game_state["coins"])
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        (x, y), distance = queue.popleft()
        if (x, y) in targets:
            return distance
        for dx, dy in _NEIGHBOUR_DELTAS:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in visited and field[nx, ny] == 0:
                visited.add((nx, ny))
                queue.append(((nx, ny), distance + 1))
    return None


def _blast_tiles(field, bomb_position: tuple[int, int]) -> set[tuple[int, int]]:
    """Return blast tiles using the framework's stone-wall stopping rule."""
    x, y = bomb_position
    tiles = {(x, y)}
    for dx, dy in _NEIGHBOUR_DELTAS:
        for distance in range(1, 4):
            nx, ny = x + dx * distance, y + dy * distance
            if field[nx, ny] == -1:
                break
            tiles.add((nx, ny))
    return tiles


def safe_crate_bomb_available(game_state: dict) -> bool:
    """Check whether dropping a bomb now can hit a crate and still be escaped."""
    field = game_state["field"]
    _, _, _, start = game_state["self"]
    if not any(field[start[0] + dx, start[1] + dy] == 1 for dx, dy in _NEIGHBOUR_DELTAS):
        return False

    blast_tiles = _blast_tiles(field, start)
    frontier = {start}
    # A newly dropped bomb has timer 4. The agent has four subsequent movement
    # opportunities before its blast is evaluated by the environment.
    for _ in range(4):
        next_frontier = set()
        for x, y in frontier:
            for dx, dy in _NEIGHBOUR_DELTAS:
                nx, ny = x + dx, y + dy
                if field[nx, ny] == 0 and (nx, ny) != start:
                    next_frontier.add((nx, ny))
        frontier = next_frontier
        if not frontier:
            return False
    return any(position not in blast_tiles for position in frontier)


def reward_from_transition(old_game_state, new_game_state, events: List[str]) -> float:
    """Combine event rewards with bounded progress toward a reachable coin."""
    reward = float(sum(EVENT_REWARDS.get(event, 0.0) for event in events))
    # On collection the next nearest coin may be much farther away; the event
    # reward already credits collection, so do not turn that into a penalty.
    if (USE_COIN_PROGRESS_SHAPING and e.COIN_COLLECTED not in events
            and old_game_state is not None and new_game_state is not None):
        old_distance = nearest_reachable_coin_distance(old_game_state)
        new_distance = nearest_reachable_coin_distance(new_game_state)
        if old_distance is not None and new_distance is not None:
            reward += COIN_PROGRESS_REWARD * max(-1, min(1, old_distance - new_distance))
    if USE_SAFE_CRATE_BOMB_SHAPING and e.BOMB_DROPPED in events:
        reward += SAFE_CRATE_BOMB_REWARD if safe_crate_bomb_available(old_game_state) else UNSAFE_OR_WASTED_BOMB_PENALTY
    return reward


def _store_transition(
    self,
    old_game_state,
    action: str,
    new_game_state,
    events: List[str],
    done: bool,
    extra_reward: float = 0.0,
    state_features=None,
    next_features=None,
) -> None:
    if old_game_state is None or action not in ACTION_TO_INDEX:
        return
    state = state_to_features(old_game_state) if state_features is None else state_features
    next_state = None if done else (state_to_features(new_game_state) if next_features is None else next_features)
    next_mask = None if done else action_mask(new_game_state)
    self.replay_buffer.add(
        state=state,
        action=ACTION_TO_INDEX[action],
        reward=reward_from_transition(old_game_state, new_game_state, events) + extra_reward,
        next_state=next_state,
        next_action_mask=next_mask,
        done=done,
    )
    self.environment_steps += 1
    self.epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * self.environment_steps / EPSILON_DECAY_STEPS)


def _optimise(self) -> None:
    if len(self.replay_buffer) < LEARNING_STARTS or self.environment_steps % TRAIN_EVERY != 0:
        return
    batch = self.replay_buffer.sample(BATCH_SIZE)
    self.q_network.train()
    predicted = self.q_network(batch.states).gather(1, batch.actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        online_next = self.q_network(batch.next_states)
        online_next[batch.next_action_masks == 0] = -torch.inf
        next_actions = online_next.argmax(dim=1, keepdim=True)
        target_next = self.target_network(batch.next_states).gather(1, next_actions).squeeze(1)
        target = batch.rewards + GAMMA * (1.0 - batch.dones) * target_next
    loss = self.loss_fn(predicted, target)
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
    self.optimizer.step()
    self.q_network.eval()
    self.gradient_steps += 1
    self.last_loss = float(loss.item())
    if self.environment_steps % TARGET_SYNC_EVERY == 0:
        self.target_network.load_state_dict(self.q_network.state_dict())


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]) -> None:
    """Store a non-terminal transition and periodically apply one DDQN update."""
    old_position = old_game_state["self"][3]
    new_position = new_game_state["self"][3]
    if not self.recent_positions or self.recent_positions[-1] != old_position:
        self.recent_positions.append(old_position)
    is_immediate_reversal = (
        USE_IMMEDIATE_REVERSAL_PENALTY
        and len(self.recent_positions) >= 2
        and new_position == self.recent_positions[-2]
        and new_position != old_position
    )
    is_short_cycle = (
        USE_SHORT_CYCLE_PENALTY
        and new_position != old_position
        and new_position in self.recent_positions
        and not any(event in _OBJECTIVE_EVENTS for event in events)
    )
    self.recent_positions.append(new_position)
    if self_action == "WAIT":
        self.consecutive_waits += 1
    else:
        self.consecutive_waits = 0
    is_consecutive_wait = (
        USE_CONSECUTIVE_WAIT_PENALTY
        and self.consecutive_waits >= CONSECUTIVE_WAIT_THRESHOLD
    )
    extra_reward = (
        (IMMEDIATE_REVERSAL_PENALTY if is_immediate_reversal else 0.0)
        + (SHORT_CYCLE_PENALTY if is_short_cycle else 0.0)
        + (CONSECUTIVE_WAIT_PENALTY if is_consecutive_wait else 0.0)
    )
    _store_transition(self, old_game_state, self_action, new_game_state, events, done=False, extra_reward=extra_reward,
                      state_features=getattr(self, "last_action_features", None),
                      next_features=state_to_features(new_game_state))
    self.last_recorded_state_step = old_game_state["step"]
    _optimise(self)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]) -> None:
    """Store terminal data only when death skipped the normal callback."""
    if last_game_state is not None and self.last_recorded_state_step != last_game_state["step"]:
        _store_transition(self, last_game_state, last_action, None, events, done=True)
        _optimise(self)
    self.completed_rounds += 1
    self.recent_positions.clear()
    self.consecutive_waits = 0
    torch.save(
        {
            "q_network": self.q_network.state_dict(),
            "feature_dim": FEATURE_DIM,
            "actions": ACTIONS,
            "architecture": ARCHITECTURE,
        },
        MODEL_FILE,
    )
    self.logger.info(
        "Round %s: replay=%s, steps=%s, updates=%s, epsilon=%.3f, loss=%s",
        self.completed_rounds, len(self.replay_buffer), self.environment_steps,
        self.gradient_steps, self.epsilon, self.last_loss,
    )
