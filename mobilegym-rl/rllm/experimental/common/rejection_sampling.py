"""
Rejection sampling utilities for multi-trajectory workflows.

This module provides rejection sampling that works with TrajectoryGroups,
tracking episode-level correctness metrics while filtering based on reward variance.
"""

from dataclasses import dataclass, field

import numpy as np

from rllm.agents.agent import Episode, TrajectoryGroup
from rllm.experimental.common.config import RejectionSamplingConfig


@dataclass
class RejectionSamplingMetrics:
    """Metrics tracked during rejection sampling."""

    # Episode-level correctness counts (for logging)
    solve_none: int = 0  # Tasks where ALL episodes are incorrect
    solve_all: int = 0  # Tasks where ALL episodes are correct
    solve_partial: int = 0  # Tasks where SOME episodes are correct

    # Group-level filtering counts
    groups_before_filter: int = 0
    groups_after_filter: int = 0
    groups_dropped_insufficient_trajs: int = 0
    dynamic_candidate_groups: int = 0
    dynamic_effective_groups: int = 0
    dynamic_rounds: int = 0
    dynamic_fallback: int = 0
    dynamic_effective_used: int = 0
    dynamic_ordinary_used: int = 0

    def reset(self):
        """Reset all metrics to zero."""
        self.solve_none = 0
        self.solve_all = 0
        self.solve_partial = 0
        self.groups_before_filter = 0
        self.groups_after_filter = 0
        self.groups_dropped_insufficient_trajs = 0
        self.dynamic_candidate_groups = 0
        self.dynamic_effective_groups = 0
        self.dynamic_rounds = 0
        self.dynamic_fallback = 0
        self.dynamic_effective_used = 0
        self.dynamic_ordinary_used = 0

    def to_dict(self, prefix: str = "batch/") -> dict:
        """Convert metrics to a dictionary for logging."""
        total_tasks = max(self.solve_none + self.solve_all + self.solve_partial, 1)

        return {
            f"{prefix}num_tasks": total_tasks,
            f"{prefix}solve_none": self.solve_none / total_tasks,
            f"{prefix}solve_all": self.solve_all / total_tasks,
            f"{prefix}solve_partial": self.solve_partial / total_tasks,
            f"{prefix}groups_before_filter": self.groups_before_filter,
            f"{prefix}groups_after_filter": self.groups_after_filter,
            f"{prefix}groups_dropped_insufficient_trajs": self.groups_dropped_insufficient_trajs,
            f"{prefix}dynamic_candidate_groups": self.dynamic_candidate_groups,
            f"{prefix}dynamic_effective_groups": self.dynamic_effective_groups,
            f"{prefix}dynamic_rounds": self.dynamic_rounds,
            f"{prefix}dynamic_fallback": self.dynamic_fallback,
            f"{prefix}dynamic_effective_used": self.dynamic_effective_used,
            f"{prefix}dynamic_ordinary_used": self.dynamic_ordinary_used,
        }


@dataclass
class RejectionSamplingState:
    """State maintained across batches for episode-level rejection sampling."""

    # Accumulated groups waiting to be processed
    accumulated_groups: list[TrajectoryGroup] = field(default_factory=list)
    # Accumulated episodes for reference
    accumulated_episodes: list[Episode] = field(default_factory=list)
    # Running metrics across accumulation
    metrics: RejectionSamplingMetrics = field(default_factory=RejectionSamplingMetrics)

    def reset(self):
        """Reset state after a batch is processed."""
        self.accumulated_groups = []
        self.accumulated_episodes = []
        self.metrics.reset()


def update_episode_metrics(
    episodes: list[Episode],
    metrics: RejectionSamplingMetrics,
) -> None:
    """
    Update episode-level metrics (solve_none/all/partial).

    Groups episodes by task_id and checks whether each episode has a
    positive trajectory-level reward.

    Args:
        episodes: List of episodes to analyze
        metrics: Metrics object to update (in-place)
    """
    # Group episodes by task_id
    episodes_by_task: dict[str, list[Episode]] = {}
    for episode in episodes:
        if len(episode.trajectories) == 0:
            continue
        # Extract task_id from episode.id format "task_id:rollout_idx"
        task_id = episode.task_id
        if task_id not in episodes_by_task:
            episodes_by_task[task_id] = []
        episodes_by_task[task_id].append(episode)

    # A positive trajectory reward represents a partial or complete solve.
    # Keep this at trajectory level rather than using step-row rewards, since
    # the episode reward is what rejection sampling uses for accumulation.
    for task_id, task_episodes in episodes_by_task.items():
        positive_reward_mask = [
            any(
                trajectory.reward is not None and float(trajectory.reward) > 0
                for trajectory in episode.trajectories
            )
            for episode in task_episodes
        ]
        if all(positive_reward_mask):
            metrics.solve_all += 1
        elif any(positive_reward_mask):
            metrics.solve_partial += 1
        else:
            metrics.solve_none += 1


def filter_groups(
    groups: list[TrajectoryGroup],
    config: RejectionSamplingConfig,
    metrics: RejectionSamplingMetrics,
) -> tuple[list[TrajectoryGroup], list[TrajectoryGroup]]:
    """
    Filter trajectory groups based on configuration.

    Args:
        groups: List of trajectory groups to filter
        config: Rejection sampling configuration
        metrics: Metrics object to update (in-place)

    Returns:
        Tuple of (filtered groups, dropped groups)
    """
    metrics.groups_before_filter += len(groups)
    filtered, dropped = [], []

    for group in groups:
        # Check minimum trajectories
        if len(group.trajectories) < config.min_trajs_per_group:
            metrics.groups_dropped_insufficient_trajs += 1
            dropped.append(group)
            continue

        filtered.append(group)

    metrics.groups_after_filter += len(filtered)
    return filtered, dropped


def is_effective_reward_group(group: TrajectoryGroup, min_reward_range: float) -> bool:
    """Return whether a complete group has non-trivial trajectory reward spread."""
    rewards = [trajectory.reward for trajectory in group.trajectories]
    if not rewards or any(reward is None or not np.isfinite(float(reward)) for reward in rewards):
        return False
    return float(np.ptp(np.asarray(rewards, dtype=float))) >= min_reward_range


def filter_episodes(
    episodes: list[Episode],
    dropped_groups: list[TrajectoryGroup],
) -> list[Episode]:
    """
    Filter episodes based on dropped groups.
    """
    # obtain a flattened list of trajectory uids (unique identifier)
    trajectory_uids = [traj.uid for group in dropped_groups for traj in group.trajectories]
    filtered_episodes = []
    for episode in episodes:
        filtered_trajectories = []
        for trajectory in episode.trajectories:
            if trajectory.uid in trajectory_uids:
                continue
            filtered_trajectories.append(trajectory)
        episode.trajectories = filtered_trajectories
        # Note: we still append even if the episode has no valid trajectories, this will be handled by the transform step
        filtered_episodes.append(episode)
    return filtered_episodes


def apply_rejection_sampling_and_filtering(
    episodes: list[Episode],
    groups: list[TrajectoryGroup],
    config: RejectionSamplingConfig,
    state: RejectionSamplingState,
) -> tuple[list[TrajectoryGroup], list[Episode], dict]:
    """
    Apply rejection sampling to trajectory groups and episodes.

    Args:
        episodes: List of episodes (for correctness metrics)
        groups: List of trajectory groups to filter
        config: Rejection sampling configuration
        state: State object for episode-level accumulation (required for "episode" mode)

    Returns:
        Tuple of (filtered groups, filtered episodes, metrics dictionary)
        filtered groups: list of trajectory groups that pass the rejection sampling criteria
        filtered episodes: list of episodes that pass the rejection sampling criteria
        metrics: metrics dictionary for logging
    """
    if config.mode == "group":
        raise NotImplementedError("Group-level rejection sampling is not implemented yet")

    metrics = state.metrics

    # Step 1: Filter groups and episodes based on config
    filtered_groups, dropped_groups = filter_groups(groups, config, metrics)
    filtered_episodes = filter_episodes(episodes, dropped_groups)

    # Step 2: Compute episode-level correctness metrics (always, for logging)
    update_episode_metrics(filtered_episodes, metrics)

    # Step 3: Apply mode-specific logic (TODO(listar2000): implement a group-level rejection sampling)
    if config.mode == "none":
        # No rejection, just return filtered groups with metrics
        return filtered_groups, filtered_episodes, metrics.to_dict()
    elif config.mode == "episode":  # Episode-level: accumulate until we have enough partial solves
        state.accumulated_groups.extend(filtered_groups)
        state.accumulated_episodes.extend(filtered_episodes)

        # Check if we have enough partial solves
        if metrics.solve_partial >= config.min_partial_solve_tasks:
            # Ready to proceed - return accumulated groups
            return state.accumulated_groups.copy(), state.accumulated_episodes.copy(), metrics.to_dict()
        else:  # Not enough - skip this batch
            return [], [], metrics.to_dict()
    else:
        raise ValueError(f"Unknown rejection sampling mode: {config.mode}")
