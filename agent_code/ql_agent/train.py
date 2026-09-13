from collections import namedtuple, deque, Counter

import pickle
import random
from typing import List

import numpy as np

import events as e
from .callbacks import ACTIONS, state_to_features, nearest_coin_distance, compute_danger_tiles, bomb_would_hit_crate

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

TRANSITION_HISTORY_SIZE = 5000  # replay buffer size -- this is now actually used for learning

ALPHA = 0.02  # learning rate (lowered: we now do many more updates per step via replay)
GAMMA = 0.9   # discount factor
REPLAY_BATCH_SIZE = 16  # extra past transitions replayed per step, on top of the fresh one

EPSILON_START = 0.3
EPSILON_MIN = 0.1  # kept higher than before so training can keep escaping bad local policies
EPSILON_DECAY = 0.997

# Custom events for reward shaping
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"
ENTERED_DANGER = "ENTERED_DANGER"
STAYED_IN_DANGER = "STAYED_IN_DANGER"
MOVED_TO_SAFETY = "MOVED_TO_SAFETY"
USELESS_BOMB = "USELESS_BOMB"
REVERSED_DIRECTION = "REVERSED_DIRECTION"
STALLING = "STALLING"

# Events we track counts of per episode, purely for diagnostics in the CSV
TRACKED_EVENTS = [
    e.COIN_COLLECTED, e.INVALID_ACTION, e.BOMB_DROPPED, e.WAITED,
    e.KILLED_SELF, e.GOT_KILLED, e.KILLED_OPPONENT,
    MOVED_TOWARDS_COIN, MOVED_AWAY_FROM_COIN,
    ENTERED_DANGER, STAYED_IN_DANGER, MOVED_TO_SAFETY, USELESS_BOMB,
    REVERSED_DIRECTION, STALLING,
]


def setup_training(self):
    """Called after `setup` in callbacks.py when training is enabled."""
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.episode_reward = 0.0
    self.episode_counts = Counter()
    self.epsilon = EPSILON_START
    self.recent_positions = deque(maxlen=2)
    self.stall_counter = 0  # steps since the agent last got closer to a coin

    header = "round,total_reward,steps,epsilon," + ",".join(TRACKED_EVENTS) + "\n"
    with open("training_rewards.csv", "w") as f:
        f.write(header)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    add_custom_events(self, old_game_state, new_game_state, events)
    self.episode_counts.update(ev for ev in events if ev in TRACKED_EVENTS)

    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)
    reward = reward_from_events(self, events)
    self.episode_reward += reward

    self.transitions.append(Transition(old_features, self_action, new_features, reward))
    update_q(self, old_features, self_action, new_features, reward)
    replay_from_buffer(self)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.episode_counts.update(ev for ev in events if ev in TRACKED_EVENTS)

    last_features = state_to_features(last_game_state)
    reward = reward_from_events(self, events)
    self.episode_reward += reward

    self.transitions.append(Transition(last_features, last_action, None, reward))
    update_q(self, last_features, last_action, None, reward)
    replay_from_buffer(self)

    counts_str = ",".join(str(self.episode_counts.get(ev, 0)) for ev in TRACKED_EVENTS)
    with open("training_rewards.csv", "a") as f:
        f.write(f"{last_game_state['round']},{self.episode_reward},{last_game_state['step']},{self.epsilon:.4f},{counts_str}\n")

    self.episode_reward = 0.0
    self.episode_counts = Counter()
    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    self.recent_positions.clear()
    self.stall_counter = 0

    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def add_custom_events(self, old_game_state, new_game_state, events):
    """Reward shaping: coin progress, bomb danger, plus an anti-oscillation
    penalty for literally walking back to the tile you were on two steps
    ago (the classic left-right / up-down flip-flop trap)."""
    old_dist = nearest_coin_distance(old_game_state)
    new_dist = nearest_coin_distance(new_game_state)
    if old_dist is not None and new_dist is not None:
        if new_dist < old_dist:
            events.append(MOVED_TOWARDS_COIN)
        elif new_dist > old_dist:
            events.append(MOVED_AWAY_FROM_COIN)

    _, _, _, old_pos = old_game_state['self']
    _, _, _, new_pos = new_game_state['self']
    old_danger = compute_danger_tiles(old_game_state['field'], old_game_state['bombs'], old_game_state['explosion_map'])
    new_danger = compute_danger_tiles(new_game_state['field'], new_game_state['bombs'], new_game_state['explosion_map'])
    was_in_danger = old_pos in old_danger
    now_in_danger = new_pos in new_danger

    if now_in_danger and was_in_danger:
        events.append(STAYED_IN_DANGER)
    elif now_in_danger and not was_in_danger:
        events.append(ENTERED_DANGER)
    elif was_in_danger and not now_in_danger:
        events.append(MOVED_TO_SAFETY)

    if e.BOMB_DROPPED in events and not bomb_would_hit_crate(old_game_state['field'], old_pos):
        events.append(USELESS_BOMB)

    # Escalating penalty for going a long time without getting any closer
    # to a coin -- this is the general-purpose safety valve against
    # getting permanently stuck, regardless of the specific numerical
    # reason a fixed point (a WAIT-preference, a 2-tile bounce, etc.)
    # formed in the first place. Any real progress resets the counter.
    if old_dist is not None and new_dist is not None and new_dist < old_dist:
        self.stall_counter = 0
    else:
        self.stall_counter += 1
    if self.stall_counter > 10:
        events.append(STALLING)

    # Anti-oscillation: did we just walk straight back to where we were
    # two steps ago? recent_positions holds [pos_two_steps_ago, pos_last_step]
    # going into this call.
    if not self.recent_positions:
        self.recent_positions.append(old_pos)
    if new_pos != old_pos:
        if len(self.recent_positions) == 2 and new_pos == self.recent_positions[0]:
            events.append(REVERSED_DIRECTION)
        self.recent_positions.append(new_pos)


def update_q(self, old_features, action, new_features, reward):
    """
    Double Q-learning update. Standard single-network Q-learning uses
    max_a' Q(s', a') as part of its own target -- since Q is a noisy
    estimate, the max operator systematically overestimates, and that
    inflated value gets bootstrapped backward into every preceding
    state via gamma. WAIT is especially prone to soaking up this bias:
    it's barely penalized directly, but still gets full credit for
    whatever good happens later in the trajectory purely through
    bootstrapping.

    Double Q-learning breaks this by using one network to pick the best
    next action and the OTHER, independently-updated network to
    evaluate it -- an overestimate in one is unlikely to be mirrored in
    the other, so it doesn't get selected as often.
    """
    if old_features is None or action not in ACTIONS:
        return
    action_idx = ACTIONS.index(action)

    # Flip a coin: update model[0] using model[1] to evaluate the next
    # state, or vice versa.
    i, j = (0, 1) if random.random() < 0.5 else (1, 0)

    old_q = self.model[i][action_idx].dot(old_features)
    if new_features is not None:
        best_next_action = int(np.argmax(self.model[i] @ new_features))
        max_next_q = self.model[j][best_next_action].dot(new_features)
    else:
        max_next_q = 0.0  # terminal state: no future reward

    td_error = reward + GAMMA * max_next_q - old_q
    self.model[i][action_idx] += ALPHA * td_error * old_features


def replay_from_buffer(self):
    """
    Re-run the Q-update on a random sample of past transitions, not just
    the one that just happened. Using only the freshest transition every
    time makes consecutive updates highly correlated, which is a known
    source of instability for linear function approximation combined with
    bootstrapping -- it can lock the policy into a self-consistent but
    wrong two-state cycle (e.g. tile A says "go right", tile B says "go
    left back to A", forever). Repeatedly re-learning from a random mix
    of older experience counteracts that by not letting any single recent
    update dominate.
    """
    if len(self.transitions) < REPLAY_BATCH_SIZE:
        return
    batch = random.sample(self.transitions, REPLAY_BATCH_SIZE)
    for transition in batch:
        update_q(self, transition.state, transition.action, transition.next_state, transition.reward)


def reward_from_events(self, events: List[str]) -> float:
    """
    Map game events (and our own custom ones) to a scalar reward.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1.0,
        e.KILLED_OPPONENT: 5.0,
        e.KILLED_SELF: -5.0,
        e.GOT_KILLED: -5.0,
        e.INVALID_ACTION: -1.0,
        e.WAITED: -0.4,
        MOVED_TOWARDS_COIN: 0.1,
        MOVED_AWAY_FROM_COIN: -0.1,
        STAYED_IN_DANGER: -0.3,
        ENTERED_DANGER: -0.3,
        MOVED_TO_SAFETY: 0.3,
        USELESS_BOMB: -0.5,
    }
    reward_sum = sum(game_rewards.get(event, 0.0) for event in events)

    if STALLING in events:
        # Grows the longer the agent goes without progress, so it
        # eventually overwhelms any fixed-size bias that's keeping it
        # stuck, no matter how that bias arose.
        reward_sum += -0.1 * (self.stall_counter - 10)

    self.logger.info(f"Awarded {reward_sum:.2f} for events {', '.join(events)}")
    return reward_sum