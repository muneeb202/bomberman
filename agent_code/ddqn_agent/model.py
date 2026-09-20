"""Small CPU-only components shared by DDQN training and inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    """A compact MLP that predicts one Q-value for every framework action."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states)


class DuelingQNetwork(nn.Module):
    """Dueling-DQN MLP with separate state-value and action-advantage heads."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(128, 1)
        self.advantage_head = nn.Linear(128, action_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        features = self.feature_layers(states)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


@dataclass(frozen=True)
class TransitionBatch:
    """A sampled, CPU-resident batch of DDQN transitions."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    next_action_masks: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Preallocated ring buffer with deterministic local RNG sampling."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int, seed: int | None = None) -> None:
        if capacity <= 0 or state_dim <= 0 or action_dim <= 0:
            raise ValueError("capacity, state_dim, and action_dim must be positive")
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self.next_action_masks = np.empty((capacity, action_dim), dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.size = 0
        self.position = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray | None,
        next_action_mask: np.ndarray | None,
        done: bool,
    ) -> None:
        """Store one transition; terminal next states are represented by zeros."""
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.state_dim,):
            raise ValueError(f"Expected state shape {(self.state_dim,)}, got {state.shape}")
        if next_state is not None:
            next_state = np.asarray(next_state, dtype=np.float32)
            if next_state.shape != (self.state_dim,):
                raise ValueError(f"Expected next_state shape {(self.state_dim,)}, got {next_state.shape}")
        if next_action_mask is not None:
            next_action_mask = np.asarray(next_action_mask, dtype=np.float32)
            if next_action_mask.shape != (self.action_dim,):
                raise ValueError(f"Expected next_action_mask shape {(self.action_dim,)}, got {next_action_mask.shape}")

        index = self.position
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = 0.0 if next_state is None else next_state
        self.next_action_masks[index] = 1.0 if next_action_mask is None else next_action_mask
        self.dones[index] = float(done)
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        """Sample unique transitions and convert them to CPU tensors."""
        if batch_size <= 0 or batch_size > self.size:
            raise ValueError(f"batch_size must be in [1, {self.size}], got {batch_size}")
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return TransitionBatch(
            states=torch.from_numpy(self.states[indices].copy()),
            actions=torch.from_numpy(self.actions[indices].copy()),
            rewards=torch.from_numpy(self.rewards[indices].copy()),
            next_states=torch.from_numpy(self.next_states[indices].copy()),
            next_action_masks=torch.from_numpy(self.next_action_masks[indices].copy()),
            dones=torch.from_numpy(self.dones[indices].copy()),
        )
