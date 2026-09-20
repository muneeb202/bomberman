from collections import namedtuple, deque, Counter
import pickle
import random
from typing import List
import numpy as np
import events as e
from .callbacks import (
    ACTIONS, state_to_features, nearest_coin_distance, nearest_crate_distance,
    nearest_opponent_distance, compute_danger_tiles, bomb_would_hit_crate,
    bomb_would_hit_opponents, is_safe_to_bomb, valid_action_mask,
)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'next_valid'))

# Replay buffer
TRANSITION_HISTORY_SIZE = 10000
REPLAY_BATCH_SIZE = 32

# Learning hyperparameters
ALPHA = 0.05          # learning rate
GAMMA = 0.95          # discount factor
EPSILON_START = 0.9   # start with heavy exploration
EPSILON_MIN = 0.05    # keep a small floor
EPSILON_DECAY = 0.999037 # decay per round -- reaches EPSILON_MIN after roughly
                       # log(EPSILON_MIN/EPSILON_START)/log(EPSILON_DECAY) rounds
                       # (~3000 rounds at these settings). Tune this to the
                       # length of the training session you're about to run:
                       # setup_training() resets epsilon to EPSILON_START at
                       # the start of every invocation, so if you train in
                       # short repeated sessions instead of one long one, each
                       # session re-explores from scratch regardless of this
                       # value -- this constant only matters within a single
                       # continuous run.

# Custom event names
MOVED_TOWARDS_COIN    = 'MOVED_TOWARDS_COIN'
MOVED_AWAY_FROM_COIN  = 'MOVED_AWAY_FROM_COIN'
MOVED_TOWARDS_CRATE   = 'MOVED_TOWARDS_CRATE'
MOVED_AWAY_FROM_CRATE = 'MOVED_AWAY_FROM_CRATE'
MOVED_TOWARDS_OPPONENT   = 'MOVED_TOWARDS_OPPONENT'
MOVED_AWAY_FROM_OPPONENT = 'MOVED_AWAY_FROM_OPPONENT'
BOMB_TARGETS_OPPONENT = 'BOMB_TARGETS_OPPONENT'
ENTERED_DANGER        = 'ENTERED_DANGER'
STAYED_IN_DANGER      = 'STAYED_IN_DANGER'
USELESS_BOMB          = 'USELESS_BOMB'
USEFUL_BOMB           = 'USEFUL_BOMB'
RISKY_BOMB_DROPPED    = 'RISKY_BOMB_DROPPED'
WAITED_UNNECESSARILY  = 'WAITED_UNNECESSARILY'
STALLING              = 'STALLING'
ESCAPED_OWN_BOMB      = 'ESCAPED_OWN_BOMB'

TRACKED_EVENTS = [
    e.COIN_COLLECTED, e.COIN_FOUND, e.CRATE_DESTROYED,
    e.INVALID_ACTION, e.BOMB_DROPPED, e.WAITED, e.KILLED_SELF,
    e.GOT_KILLED, e.KILLED_OPPONENT, e.BOMB_EXPLODED,
    MOVED_TOWARDS_COIN, MOVED_AWAY_FROM_COIN,
    MOVED_TOWARDS_CRATE, MOVED_AWAY_FROM_CRATE,
    MOVED_TOWARDS_OPPONENT, MOVED_AWAY_FROM_OPPONENT, BOMB_TARGETS_OPPONENT,
    ENTERED_DANGER, STAYED_IN_DANGER,
    USELESS_BOMB, USEFUL_BOMB, RISKY_BOMB_DROPPED, WAITED_UNNECESSARILY,
    STALLING, ESCAPED_OWN_BOMB,
]


def setup_training(self):
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.episode_reward = 0.0
    self.episode_counts = Counter()
    self.epsilon = EPSILON_START
    self.stall_counter = 0
    self.danger_counter = 0

    write_header = True
    try:
        with open('training_rewards.csv') as f:
            write_header = not f.readline().strip()
    except FileNotFoundError:
        pass
    if write_header:
        with open('training_rewards.csv', 'w') as f:
            f.write(
                'round,total_reward,steps,epsilon,'
                + ','.join(str(ev) for ev in TRACKED_EVENTS) + '\n'
            )

    self.logger.info(
        '[TRAIN] setup_training: epsilon=%.3f alpha=%.4f gamma=%.4f '
        'replay_size=%d batch=%d',
        EPSILON_START, ALPHA, GAMMA, TRANSITION_HISTORY_SIZE, REPLAY_BATCH_SIZE,
    )


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    add_custom_events(self, old_game_state, new_game_state, events)
    self.episode_counts.update(ev for ev in events if ev in TRACKED_EVENTS)

    old_features = self._last_features
    new_features = state_to_features(new_game_state, self.last_action)
    reward = reward_from_events(self, events)
    self.episode_reward += reward

    next_valid = valid_action_mask(new_game_state)
    self.transitions.append(
        Transition(old_features, self_action, new_features, reward, next_valid)
    )
    update_q(self, old_features, self_action, new_features, reward, next_valid)
    replay_from_buffer(self)

    step = new_game_state['step']
    self.logger.info(
        '[TRAIN] step=%d action=%s reward=%.3f ep_reward=%.3f events=%s',
        step, self_action, reward, self.episode_reward, events,
    )


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    if e.BOMB_EXPLODED in events and e.KILLED_SELF not in events:
        events.append(ESCAPED_OWN_BOMB)
    self.episode_counts.update(ev for ev in events if ev in TRACKED_EVENTS)
    features = self._last_features
    reward = reward_from_events(self, events)
    self.episode_reward += reward

    self.transitions.append(Transition(features, last_action, None, reward, None))
    update_q(self, features, last_action, None, reward, None)
    replay_from_buffer(self)

    counts = ','.join(str(self.episode_counts.get(ev, 0)) for ev in TRACKED_EVENTS)
    with open('training_rewards.csv', 'a') as f:
        f.write(
            f"{last_game_state['round']},{self.episode_reward:.4f},"
            f"{last_game_state['step']},{self.epsilon:.4f},{counts}\n"
        )

    self.logger.info(
        '[ROUND] #%d | reward=%.2f steps=%d epsilon=%.3f | '
        'coins=%d crates=%d killed_self=%d invalid=%d waited=%d',
        last_game_state['round'],
        self.episode_reward,
        last_game_state['step'],
        self.epsilon,
        self.episode_counts.get(e.COIN_COLLECTED, 0),
        self.episode_counts.get(e.CRATE_DESTROYED, 0),
        self.episode_counts.get(e.KILLED_SELF, 0),
        self.episode_counts.get(e.INVALID_ACTION, 0),
        self.episode_counts.get(e.WAITED, 0),
    )

    self.episode_reward = 0.0
    self.episode_counts = Counter()
    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    self.stall_counter = 0
    self.danger_counter = 0
    self.last_action = None
    self._last_features = None

    with open('my-saved-model.pt', 'wb') as f:
        pickle.dump(self.model, f)


def add_custom_events(self, old_game_state, new_game_state, events):
    """Append shaped reward events based on transition progress."""
    old_pos = old_game_state['self'][3]
    new_pos = new_game_state['self'][3]

    # ---- Coin / crate navigation progress ----
    # Only fire TOWARDS or AWAY -- never both -- so they can't cancel out.
    # Coins take priority over crates when both are visible. Towards/away
    # magnitudes are symmetric within each pair (see reward_from_events):
    # an asymmetric pair makes oscillating between two tiles a net-positive
    # strategy forever, which the agent would learn instead of ever
    # reaching the target.
    old_coin = nearest_coin_distance(old_game_state)
    new_coin = nearest_coin_distance(new_game_state)
    old_crate = nearest_crate_distance(old_game_state)
    new_crate = nearest_crate_distance(new_game_state)

    if old_coin is not None and new_coin is not None:
        if new_coin < old_coin:
            events.append(MOVED_TOWARDS_COIN)
        elif new_coin > old_coin:
            events.append(MOVED_AWAY_FROM_COIN)
        self.stall_counter = 0 if new_coin < old_coin else self.stall_counter + 1
    elif old_crate is not None and new_crate is not None:
        if new_crate < old_crate:
            events.append(MOVED_TOWARDS_CRATE)
        elif new_crate > old_crate:
            events.append(MOVED_AWAY_FROM_CRATE)
        self.stall_counter = 0 if new_crate < old_crate else self.stall_counter + 1
    else:
        self.stall_counter += 1

    # ---- Opponent navigation progress ----
    # Symmetric towards/away pair, same reasoning as coin/crate above.
    old_opp = nearest_opponent_distance(old_game_state)
    new_opp = nearest_opponent_distance(new_game_state)
    if old_opp is not None and new_opp is not None:
        if new_opp < old_opp:
            events.append(MOVED_TOWARDS_OPPONENT)
        elif new_opp > old_opp:
            events.append(MOVED_AWAY_FROM_OPPONENT)
        if old_coin is None and old_crate is None:
            self.stall_counter = 0 if new_opp < old_opp else self.stall_counter + 1

    # ---- Danger tracking ----
    # No reward for merely being safe: safety is the neutral baseline, not
    # an event to farm. A reward for "moved to safety" that's larger than
    # the penalty for "entered danger" would make toggling in and out of
    # danger a net-positive loop, the same exploit class as the coin/crate
    # shaping above.
    old_danger = compute_danger_tiles(
        old_game_state['field'], old_game_state['bombs'], old_game_state['explosion_map'],
    )
    new_danger = compute_danger_tiles(
        new_game_state['field'], new_game_state['bombs'], new_game_state['explosion_map'],
    )
    was_in_danger = old_pos in old_danger
    now_in_danger = new_pos in new_danger

    if now_in_danger and was_in_danger:
        events.append(STAYED_IN_DANGER)
        self.danger_counter += 1
    elif now_in_danger:
        events.append(ENTERED_DANGER)
        self.danger_counter = 1
    else:
        self.danger_counter = 0

    # ---- Bomb quality ----
    if e.BOMB_DROPPED in events:
        hits_crate = bomb_would_hit_crate(old_game_state['field'], old_pos)
        hits_opponent = bomb_would_hit_opponents(
            old_game_state['field'], old_pos, old_game_state['others']
        )
        safe = is_safe_to_bomb(
            old_game_state['field'], old_game_state['bombs'],
            old_game_state['others'], old_pos, old_game_state['explosion_map'],
        )
        if hits_opponent:
            events.append(BOMB_TARGETS_OPPONENT)
        if hits_crate:
            events.append(USEFUL_BOMB)
        elif not hits_opponent:
            events.append(USELESS_BOMB)
        if not safe:
            events.append(RISKY_BOMB_DROPPED)

    # ---- WAIT penalty ----
    # A WAIT is only justified when in a danger tile (waiting for the
    # explosion to clear). Any other WAIT is unnecessary.
    if e.WAITED in events and not now_in_danger:
        events.append(WAITED_UNNECESSARILY)

    # ---- Escape reward ----
    # e.BOMB_EXPLODED fires exactly when a bomb we placed goes off. If we're
    # still alive to see it, we successfully escaped -- reward that directly
    # instead of only penalising unsafe placement, since "was this bomb
    # analytically safe at drop time" and "did the agent actually walk the
    # escape route correctly over the next few steps" are different skills.
    if e.BOMB_EXPLODED in events and e.KILLED_SELF not in events:
        events.append(ESCAPED_OWN_BOMB)

    # ---- Escalating no-progress penalty ----
    # Grows the longer the agent goes without getting closer to its
    # current objective (coin > crate). This doesn't tell the agent what
    # to do -- it only makes "nothing is improving" increasingly costly.
    if self.stall_counter > 10:
        events.append(STALLING)


def update_q(self, old_features, action, new_features, reward, next_valid):
    """Double Q-learning update on a single transition."""
    if old_features is None or action not in ACTIONS:
        return
    a = ACTIONS.index(action)
    i, j = (0, 1) if random.random() < 0.5 else (1, 0)
    old_q = self.model[i][a].dot(old_features)

    if new_features is not None and next_valid is not None and np.any(next_valid):
        q_select = np.where(next_valid, self.model[i] @ new_features, -np.inf)
        best_a = int(np.argmax(q_select))
        next_q = float(self.model[j][best_a].dot(new_features))
    else:
        next_q = 0.0

    target = float(np.clip(reward + GAMMA * next_q, -20.0, 20.0))
    td_error = target - old_q
    self.model[i][a] += ALPHA * td_error * old_features


def replay_from_buffer(self):
    """Sample a mini-batch from the replay buffer."""
    if len(self.transitions) < REPLAY_BATCH_SIZE:
        return
    for t in random.sample(self.transitions, REPLAY_BATCH_SIZE):
        update_q(self, t.state, t.action, t.next_state, t.reward, t.next_valid)


def reward_from_events(self, events: List[str]) -> float:
    """
    Reward table. Towards/away pairs are symmetric by construction (see
    add_custom_events for why an asymmetric pair is exploitable).
    """
    rewards = {
        e.COIN_COLLECTED:        5.0,
        e.COIN_FOUND:            1.0,
        e.CRATE_DESTROYED:       1.0,
        e.KILLED_OPPONENT:       5.0,
        e.KILLED_SELF:          -10.0,
        e.GOT_KILLED:           -8.0,
        e.INVALID_ACTION:       -0.3,
        e.BOMB_DROPPED:          0.0,   # neutral; quality events carry the sign
        MOVED_TOWARDS_COIN:      0.4,
        MOVED_AWAY_FROM_COIN:   -0.4,
        MOVED_TOWARDS_CRATE:     0.25,
        MOVED_AWAY_FROM_CRATE:  -0.25,
        MOVED_TOWARDS_OPPONENT:  0.15,
        MOVED_AWAY_FROM_OPPONENT:-0.15,
        ENTERED_DANGER:         -0.5,
        USEFUL_BOMB:             1.5,
        USELESS_BOMB:           -1.0,
        BOMB_TARGETS_OPPONENT:   2.0,
        RISKY_BOMB_DROPPED:     -6.0,
        WAITED_UNNECESSARILY:   -0.3,
        ESCAPED_OWN_BOMB:        2.0,
    }
    total = sum(rewards.get(ev, 0.0) for ev in events)
    if STALLING in events:
        total -= 0.1 * (self.stall_counter - 10)
    if STAYED_IN_DANGER in events:
        total -= 0.5 * self.danger_counter
    self.logger.debug('[REWARD] %.3f from %s', total, events)
    return total
