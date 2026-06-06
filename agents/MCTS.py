import gymnasium as gym
import numpy as np
from copy import deepcopy
import ns_gym as nsg
import ns_gym.base as base
import random
import math

class RandomAgent(base.Agent):
    """A random agent that samples actions uniformly at random from the action space."""
    def __init__(self, env):
        self.env = env

    def act(self, observation, env):
        return self.env.action_space.sample()

class MCTSNode:
    def __init__(self, state, parent=None, action=None, prior=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}  # action -> MCTSNode
        self.visits = 0
        self.value = 0.0
        self.prior = prior

    def is_fully_expanded(self, action_space):
        return len(self.children) == action_space.n

class StandardMCTSAgent(base.Agent):
    def __init__(self, env, iterations=50, rollout_depth=20, c_param=1.41):
        self.env = env
        self.iterations = iterations
        self.rollout_depth = rollout_depth
        self.c_param = c_param

    def act(self, observation, env):
        """
        MCTS act method.
        observation: current state observation
        env: the planning environment (ns_gym planning env)
        """
        root = MCTSNode(state=observation)
        
        for _ in range(self.iterations):
            # Each iteration starts from the root
            sim_env = deepcopy(env)
            node = root
            
            # 1. Selection
            while node.is_fully_expanded(sim_env.action_space) and node.children:
                action, node = self._select_child(node)
                sim_env.step(action)
            
            # 2. Expansion
            if not node.is_fully_expanded(sim_env.action_space):
                action = self._get_unexpanded_action(node, sim_env.action_space)
                obs, reward, done, truncated, _ = sim_env.step(action)
                new_node = MCTSNode(state=obs, parent=node, action=action)
                node.children[action] = new_node
                node = new_node
            
            # 3. Simulation (Rollout)
            reward = self._simulate(sim_env)
            
            # 4. Backpropagation
            self._backpropagate(node, reward)
            
        # Return action of the child with most visits
        return self._best_action(root)

    def _select_child(self, node):
        best_val = -float('inf')
        best_action = None
        best_child = None
        for action, child in node.children.items():
            # Standard UCT formula: Q + C * sqrt(log(N_parent) / n_child)
            val = (child.value / child.visits) + self.c_param * math.sqrt(math.log(node.visits) / child.visits)
            if val > best_val:
                best_val = val
                best_action = action
                best_child = child
        return best_action, best_child

    def _get_unexpanded_action(self, node, action_space):
        actions = list(range(action_space.n))
        random.shuffle(actions)
        for a in actions:
            if a not in node.children:
                return a
        return None

    def _simulate(self, env):
        total_reward = 0
        depth = 0
        done = False
        truncated = False
        while not (done or truncated) and depth < self.rollout_depth:
            action = env.action_space.sample()
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            depth += 1
        return total_reward

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _best_action(self, root):
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action

class InformedMCTSAgent(StandardMCTSAgent):
    def __init__(self, env, mentor, noise_level=0, iterations=50, rollout_depth=20, c_param=1.41):
        super().__init__(env, iterations, rollout_depth, c_param)
        self.mentor = mentor
        self.noise_level = noise_level

    def _select_child(self, node):
        best_val = -float('inf')
        best_action = None
        best_child = None
        for action, child in node.children.items():
            # Informed UCT (AlphaGo style): Q + C * P * (sqrt(N_parent) / (1 + n_child))
            q = child.value / child.visits
            u = self.c_param * child.prior * (math.sqrt(node.visits) / (1 + child.visits))
            val = q + u
            if val > best_val:
                best_val = val
                best_action = action
                best_child = child
        return best_action, best_child

    def act(self, observation, env):
        root = MCTSNode(state=observation)
        
        for _ in range(self.iterations):
            sim_env = deepcopy(env)
            node = root
            
            # Selection
            while node.is_fully_expanded(sim_env.action_space) and node.children:
                action, node = self._select_child(node)
                sim_env.step(action)
            
            # Expansion
            if not node.is_fully_expanded(sim_env.action_space):
                # Get mentor guidance for the current node's state
                priors = self.mentor.get_guidance(node.state, self.noise_level)
                
                action = self._get_unexpanded_action(node, sim_env.action_space)
                obs, reward, done, truncated, _ = sim_env.step(action)
                
                # Create node with prior probability from mentor
                new_node = MCTSNode(state=obs, parent=node, action=action, prior=priors[action])
                node.children[action] = new_node
                node = new_node
            
            # Simulation
            reward = self._simulate(sim_env)
            
            # Backpropagation
            self._backpropagate(node, reward)
            
        return self._best_action(root)
