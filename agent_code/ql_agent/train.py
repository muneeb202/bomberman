from collections import namedtuple, deque, Counter
import pickle
import random
from typing import List
import numpy as np
import events as e
from .callbacks import ACTIONS, state_to_features, nearest_coin_distance, nearest_crate_distance, compute_danger_tiles, bomb_would_hit_crate, bomb_would_hit_opponents, is_safe_to_bomb, valid_action_mask

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'next_valid'))
TRANSITION_HISTORY_SIZE = 5000
ALPHA = 0.01
GAMMA = 0.85
REPLAY_BATCH_SIZE = 4
EPSILON_START = 0.30
EPSILON_MIN = 0.10
EPSILON_DECAY = 0.997

MOVED_TOWARDS_COIN = 'MOVED_TOWARDS_COIN'
MOVED_AWAY_FROM_COIN = 'MOVED_AWAY_FROM_COIN'
MOVED_TOWARDS_CRATE = 'MOVED_TOWARDS_CRATE'
MOVED_AWAY_FROM_CRATE = 'MOVED_AWAY_FROM_CRATE'
ENTERED_DANGER = 'ENTERED_DANGER'
STAYED_IN_DANGER = 'STAYED_IN_DANGER'
MOVED_TO_SAFETY = 'MOVED_TO_SAFETY'
USELESS_BOMB = 'USELESS_BOMB'
USEFUL_BOMB = 'USEFUL_BOMB'
BOMB_TARGETS_OPPONENT = 'BOMB_TARGETS_OPPONENT'
RISKY_BOMB_DROPPED = 'RISKY_BOMB_DROPPED'
REVERSED_DIRECTION = 'REVERSED_DIRECTION'
STALLING = 'STALLING'

TRACKED_EVENTS = [e.COIN_COLLECTED, e.COIN_FOUND, e.CRATE_DESTROYED, e.INVALID_ACTION, e.BOMB_DROPPED, e.WAITED, e.KILLED_SELF, e.GOT_KILLED, e.KILLED_OPPONENT, MOVED_TOWARDS_COIN, MOVED_AWAY_FROM_COIN, MOVED_TOWARDS_CRATE, MOVED_AWAY_FROM_CRATE, ENTERED_DANGER, STAYED_IN_DANGER, MOVED_TO_SAFETY, USELESS_BOMB, USEFUL_BOMB, BOMB_TARGETS_OPPONENT, RISKY_BOMB_DROPPED, REVERSED_DIRECTION, STALLING]


def setup_training(self):
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.episode_reward = 0.0
    self.episode_counts = Counter()
    self.epsilon = EPSILON_START
    self.recent_positions = deque(maxlen=2)
    self.stall_counter = 0

    with open('training_rewards.csv', 'w') as f:
        f.write('round,total_reward,steps,epsilon,' + ','.join(TRACKED_EVENTS) + '\n')


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    add_custom_events(self, old_game_state, new_game_state, events)
    self.episode_counts.update(ev for ev in events if ev in TRACKED_EVENTS)
    old_features, new_features = state_to_features(old_game_state), state_to_features(new_game_state)
    reward = reward_from_events(self, events)
    self.episode_reward += reward
    if new_game_state['step'] % 20 == 0 or events:
        self.logger.info('step=%s action=%s pos=%s reward=%.2f events=%s', new_game_state['step'], self_action, new_game_state['self'][3], reward, events)
    next_valid = valid_action_mask(new_game_state)
    self.transitions.append(Transition(old_features, self_action, new_features, reward, next_valid))
    update_q(self, old_features, self_action, new_features, reward, next_valid)
    replay_from_buffer(self)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.episode_counts.update(ev for ev in events if ev in TRACKED_EVENTS)
    features = state_to_features(last_game_state)
    reward = reward_from_events(self, events)
    self.episode_reward += reward
    self.transitions.append(Transition(features, last_action, None, reward, None))
    update_q(self, features, last_action, None, reward, None)
    replay_from_buffer(self)
    counts = ','.join(str(self.episode_counts.get(ev, 0)) for ev in TRACKED_EVENTS)
    with open('training_rewards.csv', 'a') as f:
        f.write(f"{last_game_state['round']},{self.episode_reward},{last_game_state['step']},{self.epsilon:.4f},{counts}\n")
    self.logger.info('ROUND %s reward=%.2f steps=%s epsilon=%.3f events=%s', last_game_state['round'], self.episode_reward, last_game_state['step'], self.epsilon, dict(self.episode_counts))
    self.episode_reward = 0.0
    self.episode_counts = Counter()
    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    self.recent_positions.clear()
    self.stall_counter = 0
    with open('my-saved-model.pt', 'wb') as f:
        pickle.dump(self.model, f)


def add_custom_events(self, old_game_state, new_game_state, events):
    old_coin, new_coin = nearest_coin_distance(old_game_state), nearest_coin_distance(new_game_state)
    old_crate, new_crate = nearest_crate_distance(old_game_state), nearest_crate_distance(new_game_state)

    # Always use coin progress when a coin is reachable. Only fall back to crates
    # when there is no reachable coin.
    if old_coin is not None and new_coin is not None:
        if new_coin < old_coin: events.append(MOVED_TOWARDS_COIN)
        elif new_coin > old_coin: events.append(MOVED_AWAY_FROM_COIN)
        self.stall_counter = 0 if new_coin < old_coin else self.stall_counter + 1
    elif old_crate is not None and new_crate is not None:
        if new_crate < old_crate: events.append(MOVED_TOWARDS_CRATE)
        elif new_crate > old_crate: events.append(MOVED_AWAY_FROM_CRATE)
        self.stall_counter = 0 if new_crate < old_crate else self.stall_counter + 1
    else:
        self.stall_counter += 1

    old_pos, new_pos = old_game_state['self'][3], new_game_state['self'][3]
    old_danger = compute_danger_tiles(old_game_state['field'], old_game_state['bombs'], old_game_state['explosion_map'])
    new_danger = compute_danger_tiles(new_game_state['field'], new_game_state['bombs'], new_game_state['explosion_map'])
    was_danger, now_danger = old_pos in old_danger, new_pos in new_danger
    if now_danger and was_danger: events.append(STAYED_IN_DANGER)
    elif now_danger: events.append(ENTERED_DANGER)
    elif was_danger: events.append(MOVED_TO_SAFETY)

    if e.BOMB_DROPPED in events:
        hits_crate = bomb_would_hit_crate(old_game_state['field'], old_pos)
        hits_opponent = bomb_would_hit_opponents(old_game_state['field'], old_pos, old_game_state['others'])
        if hits_crate: events.append(USEFUL_BOMB)
        if hits_opponent: events.append(BOMB_TARGETS_OPPONENT)
        if not hits_crate and not hits_opponent: events.append(USELESS_BOMB)
        if not is_safe_to_bomb(old_game_state['field'], old_game_state['bombs'], old_game_state['others'], old_pos): events.append(RISKY_BOMB_DROPPED)

    if self.stall_counter > 10: events.append(STALLING)

    if not self.recent_positions: self.recent_positions.append(old_pos)
    if new_pos != old_pos:
        if len(self.recent_positions) == 2 and new_pos == self.recent_positions[0]: events.append(REVERSED_DIRECTION)
        self.recent_positions.append(new_pos)


def update_q(self, old_features, action, new_features, reward, next_valid):
    if old_features is None or action not in ACTIONS: return
    a = ACTIONS.index(action)
    i, j = (0, 1) if random.random() < 0.5 else (1, 0)
    old_q = self.model[i][a].dot(old_features)
    if new_features is not None and next_valid is not None and np.any(next_valid):
        q_next = np.where(next_valid, self.model[i] @ new_features, -np.inf)
        best = int(np.argmax(q_next))
        next_q = self.model[j][best].dot(new_features)
    else:
        next_q = 0.0
    target = float(np.clip(reward + GAMMA * next_q, -15.0, 15.0))
    self.model[i][a] += ALPHA * (target - old_q) * old_features


def replay_from_buffer(self):
    if len(self.transitions) < REPLAY_BATCH_SIZE: return
    for t in random.sample(self.transitions, REPLAY_BATCH_SIZE):
        update_q(self, t.state, t.action, t.next_state, t.reward, t.next_valid)


def reward_from_events(self, events: List[str]) -> float:
    rewards = {
        e.COIN_COLLECTED: 1.0, e.COIN_FOUND: 0.2, e.CRATE_DESTROYED: 0.1,
        e.KILLED_OPPONENT: 5.0, e.KILLED_SELF: -5.0, e.GOT_KILLED: -5.0,
        e.INVALID_ACTION: -1.0, e.WAITED: -0.4,
        MOVED_TOWARDS_COIN: 0.15, MOVED_AWAY_FROM_COIN: -0.12,
        MOVED_TOWARDS_CRATE: 0.07, MOVED_AWAY_FROM_CRATE: -0.07,
        STAYED_IN_DANGER: -0.3, ENTERED_DANGER: -0.3, MOVED_TO_SAFETY: 0.3,
        USELESS_BOMB: -0.5, USEFUL_BOMB: 1.0, BOMB_TARGETS_OPPONENT: 1.0,
        RISKY_BOMB_DROPPED: -1.0, REVERSED_DIRECTION: -0.5,
    }
    total = sum(rewards.get(ev, 0.0) for ev in events)
    if STALLING in events: total -= 0.1 * (self.stall_counter - 10)
    self.logger.debug('reward=%.2f events=%s', total, events)
    return total
