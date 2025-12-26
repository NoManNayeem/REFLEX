"""
Advanced Reinforcement Learning Components
Implements PPO, Prioritized Experience Replay, and advanced RL algorithms
Based on latest research and best practices (2025)
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import os

logger = logging.getLogger(__name__)


class SumTree:
    """
    Sum Tree data structure for efficient prioritized experience replay.
    Based on Schaul et al. (2015) "Prioritized Experience Replay"
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Binary tree structure
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
        
    def _propagate(self, idx: int, change: float):
        """Update parent nodes with change"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Find sample on leaf node"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def add(self, priority: float, data: Any):
        """Add sample with priority"""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, priority: float):
        """Update priority of sample"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, Any, float]:
        """Get sample by value"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.data[data_idx], self.tree[idx])
    
    @property
    def total(self) -> float:
        """Total priority"""
        return self.tree[0]


@dataclass
class PPOParams:
    """PPO hyperparameters"""
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    Samples trajectories based on TD-error or advantage magnitude
    """
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        """
        Args:
            capacity: Maximum number of trajectories
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (starts at beta, increases to 1.0)
            beta_increment: How much to increment beta per sample
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.position = 0
        
    def add(self, trajectory: Dict[str, Any], priority: Optional[float] = None):
        """Add trajectory with priority"""
        if priority is None:
            # Use advantage magnitude or TD-error as priority
            priority = abs(trajectory.get('advantage', trajectory.get('reward', 1.0)))
            priority = max(priority, 1e-6)  # Minimum priority
        
        # Scale by alpha
        priority = (priority + 1e-6) ** self.alpha
        self.max_priority = max(self.max_priority, priority)
        
        self.tree.add(priority, trajectory)
    
    def sample(self, batch_size: int) -> Tuple[List[Dict[str, Any]], List[int], np.ndarray]:
        """
        Sample batch of trajectories
        Returns: (batch, indices, importance_weights)
        """
        batch = []
        indices = []
        priorities = []
        
        segment = self.tree.total / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, data, priority = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)
        
        # Compute importance sampling weights
        probabilities = np.array(priorities) / self.tree.total
        weights = (self.capacity * probabilities) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return batch, indices, weights
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for sampled trajectories"""
        for idx, priority in zip(indices, priorities):
            priority = (abs(priority) + 1e-6) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return self.tree.n_entries


class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) Trainer
    Implements PPO with GAE (Generalized Advantage Estimation)
    Based on Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
    """
    
    def __init__(self, params: PPOParams = None):
        self.params = params or PPOParams()
        logger.info(f"Initialized PPO Trainer with params: {self.params}")
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        next_value: float = 0.0,
        dones: Optional[List[bool]] = None
    ) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            next_value: Value of next state (0 if terminal)
            dones: List of done flags
            
        Returns:
            advantages: List of advantages
            returns: List of returns (discounted rewards)
        """
        if dones is None:
            dones = [False] * len(rewards)
        
        advantages = []
        returns = []
        gae = 0
        
        # Compute backwards
        for step in reversed(range(len(rewards))):
            if dones[step]:
                delta = rewards[step] - values[step]
                gae = delta
            else:
                delta = rewards[step] + self.params.gamma * (next_value if step == len(rewards) - 1 else values[step + 1]) - values[step]
                gae = delta + self.params.gamma * self.params.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
            next_value = values[step]
        
        return advantages, returns
    
    def compute_advantages(
        self,
        trajectories: List[Dict[str, Any]],
        use_gae: bool = True
    ) -> List[float]:
        """
        Compute advantages for trajectories using PPO-style computation
        
        Args:
            trajectories: List of trajectory dictionaries
            use_gae: Whether to use GAE or simple advantage
            
        Returns:
            List of advantages
        """
        if not trajectories:
            return []
        
        rewards = [t.get('reward', 0.0) for t in trajectories]
        
        if use_gae:
            # Estimate values (can be improved with value network)
            values = [t.get('value_estimate', t.get('reward', 0.0)) for t in trajectories]
            
            # Compute GAE
            advantages, returns = self.compute_gae(rewards, values)
            
            # Normalize advantages
            advantages = np.array(advantages)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            return advantages.tolist()
        else:
            # Simple advantage: reward - baseline
            rewards = np.array(rewards)
            baseline = rewards.mean()
            advantages = rewards - baseline
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return advantages.tolist()
    
    def compute_ppo_loss(
        self,
        old_log_probs: List[float],
        new_log_probs: List[float],
        advantages: List[float],
        values: List[float],
        returns: List[float]
    ) -> Dict[str, float]:
        """
        Compute PPO loss with clipping
        
        Args:
            old_log_probs: Old policy log probabilities
            new_log_probs: New policy log probabilities
            advantages: Advantage estimates
            values: Value estimates
            returns: Returns (target values)
            
        Returns:
            Dictionary with loss components
        """
        old_log_probs = np.array(old_log_probs)
        new_log_probs = np.array(new_log_probs)
        advantages = np.array(advantages)
        values = np.array(values)
        returns = np.array(returns)
        
        # Compute ratio
        ratio = np.exp(new_log_probs - old_log_probs)
        
        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1 - self.params.clip_epsilon, 1 + self.params.clip_epsilon) * advantages
        policy_loss = -np.minimum(surr1, surr2).mean()
        
        # Value loss
        value_loss = 0.5 * ((values - returns) ** 2).mean()
        
        # Entropy bonus (for exploration)
        entropy = -(np.exp(new_log_probs) * new_log_probs).mean()
        
        # Total loss
        total_loss = policy_loss + self.params.value_coef * value_loss - self.params.entropy_coef * entropy
        
        return {
            'total_loss': float(total_loss),
            'policy_loss': float(policy_loss),
            'value_loss': float(value_loss),
            'entropy': float(entropy),
            'mean_ratio': float(ratio.mean())
        }


class AdaptiveRewardWeights:
    """
    Adaptive reward weighting system
    Learns optimal weights for different reward components
    """
    
    def __init__(self, initial_weights: Optional[Dict[str, float]] = None):
        """
        Args:
            initial_weights: Initial weight configuration
        """
        if initial_weights is None:
            initial_weights = {
                'task_success': 0.35,
                'quality_score': 0.25,
                'efficiency_score': 0.1,
                'user_feedback': 0.1,
                'critic_score': 0.2
            }
        
        self.weights = initial_weights.copy()
        self.weight_history = []
        self.performance_history = []
        self.learning_rate = 0.01
        
        logger.info(f"Initialized adaptive reward weights: {self.weights}")
    
    def compute_reward(
        self,
        task_success: float,
        quality_score: float,
        efficiency_score: float,
        user_feedback: float,
        critic_score: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute weighted reward
        
        Returns:
            (total_reward, component_contributions)
        """
        components = {
            'task_success': task_success,
            'quality_score': quality_score,
            'efficiency_score': efficiency_score,
            'user_feedback': user_feedback,
            'critic_score': critic_score
        }
        
        total = sum(self.weights[k] * v for k, v in components.items())
        
        contributions = {k: self.weights[k] * v for k, v in components.items()}
        
        return total, contributions
    
    def update_weights(
        self,
        performance_metric: float,
        component_performances: Optional[Dict[str, float]] = None
    ):
        """
        Update weights based on performance
        
        Args:
            performance_metric: Overall performance (higher is better)
            component_performances: Performance of individual components
        """
        self.performance_history.append(performance_metric)
        
        if len(self.performance_history) < 10:
            return  # Need more data
        
        # Simple gradient-based update
        # Increase weights for components that correlate with performance
        if component_performances:
            for component, perf in component_performances.items():
                if component in self.weights:
                    # Update weight based on correlation
                    gradient = self.learning_rate * perf
                    self.weights[component] = max(0.0, min(1.0, self.weights[component] + gradient))
        
        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        
        self.weight_history.append(self.weights.copy())
        logger.debug(f"Updated reward weights: {self.weights}")
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights"""
        return self.weights.copy()
    
    def save(self, path: str):
        """Save weights to file"""
        data = {
            'weights': self.weights,
            'weight_history': self.weight_history[-100:],  # Last 100 updates
            'performance_history': self.performance_history[-100:]
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved reward weights to {path}")
    
    def load(self, path: str):
        """Load weights from file"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                self.weights = data.get('weights', self.weights)
                self.weight_history = data.get('weight_history', [])
                self.performance_history = data.get('performance_history', [])
                logger.info(f"Loaded reward weights from {path}")
        else:
            logger.warning(f"Weights file not found: {path}")


class GSPOTrainer:
    """
    Group Relative Policy Optimization (GSPO) Trainer
    Groups trajectories by similarity and computes relative advantages
    Based on recent research (2024-2025)
    """
    
    def __init__(self, num_groups: int = 5):
        """
        Args:
            num_groups: Number of groups to cluster trajectories into
        """
        self.num_groups = num_groups
        logger.info(f"Initialized GSPO Trainer with {num_groups} groups")
    
    def cluster_trajectories(
        self,
        trajectories: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Cluster trajectories by similarity
        
        Uses simple k-means on reward and skill features
        """
        if len(trajectories) < self.num_groups:
            return [trajectories]
        
        # Extract features for clustering
        features = []
        for traj in trajectories:
            feat = [
                traj.get('reward', 0.0),
                len(traj.get('relevant_skills', [])),
                traj.get('critic_score', 0.0)
            ]
            features.append(feat)
        
        features = np.array(features)
        
        # Simple k-means clustering
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(self.num_groups, len(trajectories)), random_state=42)
            labels = kmeans.fit_predict(features)
        except ImportError:
            # Fallback to simple distance-based clustering
            logger.warning("sklearn not available, using simple clustering")
            # Simple distance-based grouping
            labels = []
            centroids = []
            for i, feat in enumerate(features):
                if i < self.num_groups:
                    labels.append(i)
                    centroids.append(feat)
                else:
                    # Find closest centroid
                    distances = [np.linalg.norm(feat - c) for c in centroids]
                    labels.append(np.argmin(distances))
        
        # Group trajectories
        groups = [[] for _ in range(self.num_groups)]
        for traj, label in zip(trajectories, labels):
            groups[label].append(traj)
        
        # Remove empty groups
        groups = [g for g in groups if g]
        
        logger.debug(f"Clustered {len(trajectories)} trajectories into {len(groups)} groups")
        return groups
    
    def compute_group_relative_advantages(
        self,
        groups: List[List[Dict[str, Any]]]
    ) -> List[float]:
        """
        Compute advantages relative to group means
        """
        all_advantages = []
        
        for group in groups:
            if not group:
                continue
            
            rewards = [t.get('reward', 0.0) for t in group]
            group_mean = np.mean(rewards)
            group_std = np.std(rewards) + 1e-8
            
            # Compute relative advantages within group
            for traj in group:
                reward = traj.get('reward', 0.0)
                advantage = (reward - group_mean) / group_std
                all_advantages.append(advantage)
        
        return all_advantages

