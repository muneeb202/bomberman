"""
SARSA Agent - train.py
======================
One-step linear SARSA, following Ngo (2021) eq. (4):

    Q(s_t, a_t) <- Q(s_t, a_t) + alpha * [ r_t+1 + gamma * Q(s_t+1, a_t+1)
                                           - Q(s_t, a_t) ]

Unlike Q-learning, the target bootstraps from the action the policy actually
takes at s' rather than the best available one, so SARSA optimises the policy
it follows, exploration noise included. That makes it less prone to walking
into catastrophic states it "would have escaped optimally".

Why the update is deferred by one step
--------------------------------------
The environment loop is:

    act(s_t)                                   -> a_t
    game_events_occurred(s_t, a_t, s_t+1, ev)

so when game_events_occurred runs for step t, act() has not yet been called
on s_t+1 and a_t+1 does not exist. An on-policy target needs it. We therefore
hold (phi(s_t), a_t, r_t) and apply the update on the *next* callback, when
a_t+1 arrives as that call's `self_action`. end_of_round flushes the last
held transition and then applies the terminal update with Q(s', .) = 0.

No eligibility traces
---------------------
The reference uses plain one-step SARSA. Traces are also what made the
previous version unstable here: decaying at gamma*lambda = 0.76 with
alpha = 0.05 gives an effective step size near 0.21 applied to every action
row at once, which diverges quickly with shaped rewards of this magnitude.

No replay buffer
----------------
SARSA is on-policy; replaying tuples generated under an older epsilon would
bias the target.
"""

from collections import Counter
import pickle
from typing import List

import numpy as np
import events as e

from .callbacks import (
    ACTIONS, state_to_features,
    nearest_coin_distance, nearest_crate_distance,
    compute_danger_tiles, bomb_would_hit_crate,
    bomb_would_hit_opponents, is_safe_to_bomb,
    valid_action_mask, bfs,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
# Matched to ql_agent so the ONLY difference between the two models is the
# update rule (SARSA's on-policy Q(s',a') vs Q-learning's max_a' Q(s',a')).
# That keeps the comparison in the report a clean one-variable experiment.
ALPHA          = 0.05
GAMMA          = 0.95
EPSILON_START  = 0.9
EPSILON_MIN    = 0.05
EPSILON_DECAY  = 0.995

# ---------------------------------------------------------------------------
# Custom event names (identical set to ql_agent for easy comparison)
# ---------------------------------------------------------------------------
MOVED_TOWARDS_COIN       = 'MOVED_TOWARDS_COIN'
MOVED_AWAY_FROM_COIN     = 'MOVED_AWAY_FROM_COIN'
MOVED_TOWARDS_CRATE      = 'MOVED_TOWARDS_CRATE'
MOVED_AWAY_FROM_CRATE    = 'MOVED_AWAY_FROM_CRATE'
ENTERED_DANGER           = 'ENTERED_DANGER'
STAYED_IN_DANGER         = 'STAYED_IN_DANGER'
MOVED_TO_SAFETY          = 'MOVED_TO_SAFETY'
USELESS_BOMB             = 'USELESS_BOMB'
USEFUL_BOMB              = 'USEFUL_BOMB'
BOMB_TARGETS_OPPONENT    = 'BOMB_TARGETS_OPPONENT'
RISKY_BOMB_DROPPED       = 'RISKY_BOMB_DROPPED'
WAITED_UNNECESSARILY     = 'WAITED_UNNECESSARILY'
MOVED_TOWARDS_OPPONENT   = 'MOVED_TOWARDS_OPPONENT'
MOVED_AWAY_FROM_OPPONENT = 'MOVED_AWAY_FROM_OPPONENT'
STALLING                 = 'STALLING'

TRACKED_EVENTS = [
    e.COIN_COLLECTED, e.COIN_FOUND, e.CRATE_DESTROYED,
    e.INVALID_ACTION, e.BOMB_DROPPED, e.WAITED,
    e.KILLED_SELF, e.GOT_KILLED, e.KILLED_OPPONENT,
    MOVED_TOWARDS_COIN, MOVED_AWAY_FROM_COIN,
    MOVED_TOWARDS_CRATE, MOVED_AWAY_FROM_CRATE,
    ENTERED_DANGER, STAYED_IN_DANGER, MOVED_TO_SAFETY,
    USELESS_BOMB, USEFUL_BOMB, BOMB_TARGETS_OPPONENT,
    RISKY_BOMB_DROPPED, WAITED_UNNECESSARILY,
    MOVED_TOWARDS_OPPONENT, MOVED_AWAY_FROM_OPPONENT,
    STALLING,
]


# ---------------------------------------------------------------------------
# Training lifecycle
# ---------------------------------------------------------------------------

def setup_training(self):
    """Called once after setup() at the start of training."""
    self.episode_reward = 0.0
    self.episode_counts = Counter()
    self.epsilon        = EPSILON_START
    self.stall_counter  = 0
    self.danger_counter = 0

    # Held transition (phi(s_t), a_t, r_t), applied once a_t+1 is known.
    self._pending = None

    with open('training_rewards.csv', 'w') as f:
        f.write(
            'round,total_reward,steps,epsilon,'
            + ','.join(str(ev) for ev in TRACKED_EVENTS) + '\n'
        )
    self.logger.info(
        '[TRAIN] SARSA setup: epsilon=%.3f alpha=%.4f gamma=%.4f',
        EPSILON_START, ALPHA, GAMMA,
    )


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    """
    Called at every step during training.

    This call gives us (s_t, a_t, r_t, s_t+1) but NOT a_t+1, which the
    on-policy target needs. So we complete the transition held from step
    t-1 - whose s' is this call's old_game_state and whose a' is this
    call's self_action - and then hold the current one.
    """
    add_custom_events(self, old_game_state, new_game_state, events)
    self.episode_counts.update(ev for ev in events if ev in TRACKED_EVENTS)

    reward = reward_from_events(self, events)
    self.episode_reward += reward

    # phi(s_t): computed straight from the state we were given.
    features = state_to_features(old_game_state)

    # Complete the previous step: (phi(s_t-1), a_t-1, r_t-1) + (phi(s_t), a_t)
    if self._pending is not None:
        prev_features, prev_action, prev_reward = self._pending
        update_sarsa(self, prev_features, prev_action, features, self_action, prev_reward)

    # Hold this step until a_t+1 shows up on the next callback.
    self._pending = (features, self_action, reward)

    step    = new_game_state['step']
    old_pos = old_game_state['self'][3]
    new_pos = new_game_state['self'][3]
    danger_events = [
        ev for ev in events
        if ev in (ENTERED_DANGER, STAYED_IN_DANGER, MOVED_TO_SAFETY,
                  RISKY_BOMB_DROPPED, e.KILLED_SELF, e.GOT_KILLED)
    ]
    self.logger.info(
        '[TRAIN] step=%d %s->%s action=%s reward=%.3f ep_reward=%.3f events=%s',
        step, old_pos, new_pos, self_action, reward, self.episode_reward, events,
    )
    if danger_events:
        old_danger = compute_danger_tiles(
            old_game_state['field'], old_game_state['bombs'], old_game_state['explosion_map'],
        )
        new_danger = compute_danger_tiles(
            new_game_state['field'], new_game_state['bombs'], new_game_state['explosion_map'],
        )
        self.logger.warning(
            '[DANGER] step=%d action=%s danger_events=%s | '
            'old_pos=%s in_danger=%s | new_pos=%s in_danger=%s',
            step, self_action, danger_events,
            old_pos, old_pos in old_danger,
            new_pos, new_pos in new_danger,
        )


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Two updates happen here: the transition still held from the previous
    step is completed using this final state/action as its (s', a'), and
    then the final step itself is applied with Q(s', .) = 0.
    """
    self.episode_counts.update(ev for ev in events if ev in TRACKED_EVENTS)
    reward = reward_from_events(self, events)
    self.episode_reward += reward

    features = state_to_features(last_game_state)

    # Flush the held transition, with (phi(s_T), a_T) as its s'/a'.
    if self._pending is not None:
        prev_features, prev_action, prev_reward = self._pending
        update_sarsa(self, prev_features, prev_action, features, last_action, prev_reward)

    # Terminal update for the final step: no future state to bootstrap from.
    update_sarsa(self, features, last_action, None, None, reward)
    self._pending = None

    # Logging
    counts = ','.join(str(self.episode_counts.get(ev, 0)) for ev in TRACKED_EVENTS)
    with open('training_rewards.csv', 'a') as f:
        f.write(
            f"{last_game_state['round']},{self.episode_reward:.4f},"
            f"{last_game_state['step']},{self.epsilon:.4f},{counts}\n"
        )
    self.logger.info(
        '[ROUND] #%d | reward=%.2f steps=%d epsilon=%.3f | '
        'coins=%d crates=%d killed_self=%d got_killed=%d risky_bombs=%d invalid=%d waited=%d',
        last_game_state['round'],
        self.episode_reward,
        last_game_state['step'],
        self.epsilon,
        self.episode_counts.get(e.COIN_COLLECTED, 0),
        self.episode_counts.get(e.CRATE_DESTROYED, 0),
        self.episode_counts.get(e.KILLED_SELF, 0),
        self.episode_counts.get(e.GOT_KILLED, 0),
        self.episode_counts.get(RISKY_BOMB_DROPPED, 0),
        self.episode_counts.get(e.INVALID_ACTION, 0),
        self.episode_counts.get(e.WAITED, 0),
    )
    self.logger.info('[ROUND] #%d full_counts=%s',
                     last_game_state['round'], dict(self.episode_counts))

    self.episode_reward = 0.0
    self.episode_counts = Counter()
    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    self.stall_counter  = 0
    self.danger_counter = 0

    with open('my-saved-model.pt', 'wb') as f:
        pickle.dump(self.model, f)


# ---------------------------------------------------------------------------
# SARSA core update
# ---------------------------------------------------------------------------

def update_sarsa(self, features, action, next_features, next_action, reward):
    """
    One-step linear SARSA update (Ngo eq. 4):

        w[a] += alpha * [ r + gamma * Q(s', a') - Q(s, a) ] * phi(s)

    Only the row for the action actually taken is updated. `next_action` is
    the action the policy genuinely chose at s', which is what makes this
    on-policy; passing None for it (terminal state) drops the bootstrap term.
    """
    if features is None or action not in ACTIONS:
        return

    a_idx = ACTIONS.index(action)
    q_sa = float(self.model[a_idx].dot(features))

    if next_features is not None and next_action in ACTIONS:
        q_next = float(self.model[ACTIONS.index(next_action)].dot(next_features))
    else:
        q_next = 0.0

    target = float(np.clip(reward + GAMMA * q_next, -20.0, 20.0))
    td_error = target - q_sa

    self.model[a_idx] += ALPHA * td_error * features

    self.logger.debug(
        '[SARSA] a=%s a_next=%s q_sa=%.3f target=%.3f delta=%.3f',
        action, next_action, q_sa, target, td_error,
    )


# ---------------------------------------------------------------------------
# Custom event shaping  (identical logic to ql_agent for a fair comparison)
# ---------------------------------------------------------------------------

def add_custom_events(self, old_game_state, new_game_state, events):
    """Append shaped reward events based on transition progress."""
    old_pos = old_game_state['self'][3]
    new_pos = new_game_state['self'][3]

    # ---- Coin / crate navigation progress ----
    old_coin  = nearest_coin_distance(old_game_state)
    new_coin  = nearest_coin_distance(new_game_state)
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
    def _nearest_opp_dist(gs):
        opps = gs['others']
        if not opps:
            return None
        danger = compute_danger_tiles(gs['field'], gs['bombs'], gs['explosion_map'])
        d, _ = bfs(gs['field'], gs['bombs'], gs['others'], gs['self'][3], danger)
        vals = [d[p] for _, _, _, p in opps if p in d]
        return min(vals) if vals else None

    old_opp = _nearest_opp_dist(old_game_state)
    new_opp = _nearest_opp_dist(new_game_state)
    if old_opp is not None and new_opp is not None:
        if new_opp < old_opp:
            events.append(MOVED_TOWARDS_OPPONENT)
        elif new_opp > old_opp:
            events.append(MOVED_AWAY_FROM_OPPONENT)
        if old_coin is None and old_crate is None:
            self.stall_counter = 0 if new_opp < old_opp else self.stall_counter + 1

    # ---- Danger tracking ----
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
    elif was_in_danger:
        events.append(MOVED_TO_SAFETY)
        self.danger_counter = 0

    # ---- Bomb quality ----
    if e.BOMB_DROPPED in events:
        hits_crate   = bomb_would_hit_crate(old_game_state['field'], old_pos)
        hits_opponent = bomb_would_hit_opponents(
            old_game_state['field'], old_pos, old_game_state['others']
        )
        safe = is_safe_to_bomb(
            old_game_state['field'], old_game_state['bombs'],
            old_game_state['others'], old_pos, old_game_state['explosion_map'],
        )
        if hits_crate:
            events.append(USEFUL_BOMB)
        if hits_opponent:
            events.append(BOMB_TARGETS_OPPONENT)
        if not hits_crate and not hits_opponent:
            events.append(USELESS_BOMB)
        if not safe:
            events.append(RISKY_BOMB_DROPPED)
        self.logger.info(
            '[BOMB] pos=%s hits_crate=%s hits_opp=%s safe=%s -> %s',
            old_pos, hits_crate, hits_opponent, safe,
            [ev for ev in events if ev in (USEFUL_BOMB, USELESS_BOMB,
                                           BOMB_TARGETS_OPPONENT, RISKY_BOMB_DROPPED)],
        )

    # ---- Unnecessary WAIT ----
    if e.WAITED in events and not now_in_danger:
        events.append(WAITED_UNNECESSARILY)

    # ---- Escalating stall penalty ----
    if self.stall_counter > 10:
        events.append(STALLING)


# ---------------------------------------------------------------------------
# Reward table
# ---------------------------------------------------------------------------

def reward_from_events(self, events: List[str]) -> float:
    """
    Identical to ql_agent's reward table, deliberately.

    The paper's Table 1 (sparse: move -1, wall +30, kill +500, die -300) was
    tried here and did not work at this scale. Ngo trained 100 generations of
    10,000 episodes on a 7x7 board; this project runs ~1-2k rounds on 17x17.
    With rewards that sparse the agent essentially never stumbles into
    "bomb -> escape -> crate destroyed", so the only signal it reliably gets
    is the death penalty, and it learns that every action everywhere leads to
    death (observed Q values sat at -230 to -290, i.e. the -300 death term
    lightly discounted).

    Keeping this table identical to ql_agent also makes the comparison in the
    report a one-variable experiment: same features, same rewards, same gamma
    and epsilon schedule, differing only in SARSA's on-policy target.
    """
    rewards = {
        # Core game outcomes
        e.COIN_COLLECTED:         3.0,
        e.COIN_FOUND:             0.5,
        e.CRATE_DESTROYED:        1.0,
        e.KILLED_OPPONENT:        5.0,
        e.KILLED_SELF:           -10.0,
        e.GOT_KILLED:             -8.0,
        e.INVALID_ACTION:         -0.3,
        e.WAITED:                 -0.5,
        e.BOMB_DROPPED:            0.0,
        # Navigation shaping
        MOVED_TOWARDS_COIN:       0.5,
        MOVED_AWAY_FROM_COIN:    -0.3,
        MOVED_TOWARDS_CRATE:      0.4,
        MOVED_AWAY_FROM_CRATE:   -0.2,
        MOVED_TOWARDS_OPPONENT:   0.3,
        MOVED_AWAY_FROM_OPPONENT:-0.1,
        # Danger / safety
        STAYED_IN_DANGER:        -2.0,
        ENTERED_DANGER:          -0.5,
        MOVED_TO_SAFETY:          1.5,
        # Bomb quality
        USELESS_BOMB:            -1.5,
        USEFUL_BOMB:              2.0,
        BOMB_TARGETS_OPPONENT:    2.5,
        RISKY_BOMB_DROPPED:      -6.0,
        # Idle penalty
        WAITED_UNNECESSARILY:    -0.8,
    }
    total = sum(rewards.get(ev, 0.0) for ev in events)
    if STALLING in events:
        total -= 0.1 * (self.stall_counter - 10)
    if STAYED_IN_DANGER in events:
        total -= 0.5 * self.danger_counter
    self.logger.debug('[REWARD] %.3f from %s', total, events)
    return total