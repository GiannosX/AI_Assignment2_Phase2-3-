import math
import random
from collections import namedtuple, deque
import ns_gym.base as base
import numpy as np
from ns_gym.benchmark_algorithms.algo_utils import observation_type_checker, reward_type_checker
import torch
import torch.nn as nn
import time
import torch.optim as optim
import torch.nn.functional as F
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def network_input_checker(x, state_size):
    """Make sure the input to the neural network is in the correct format

    Args:
        x (Union[int, np.ndarray, dict]): Input to the neural network.
        state_size (int): Size of the state space.
    Returns:
        torch.Tensor: Tensor to be fed into the neural network.
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    if isinstance(x, dict) and 'state' in x:
        x = x['state']

    s = x 
    if type(s) == int:
        arr = torch.zeros(state_size)
        arr[s] = 1
        arr = arr.unsqueeze(0)
        s = arr

    if not isinstance(s, torch.Tensor):
        s = torch.Tensor(x)

    s = s.to(device)
    return s


class ReplayBuffer(object):
    """Replay buffer to store and sample experience tuples.
    """
    def __init__(self, capacity,state_size, action_size) -> None:
        """Initialize a ReplayBuffer object.
        
        Args:
            capacity (int): maximum size of buffer
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.state_size = state_size
        self.action_size = action_size

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def add(self, state, action, reward, next_state,done):
        """Add a new experience to memory.

        Args:
            state (Union[int, np.ndarray, dict]): current state
            action (int): action taken
            reward (float): reward received
            next_state (Union[int, np.ndarray, dict]): next state
            done (bool): whether the episode has ended
        """

        if isinstance(state, dict) and 'state' in state:
            state = state['state']
        if isinstance(next_state, base.Reward):
            reward = reward.reward

        state = network_input_checker(state,state_size=self.state_size)
        next_state = network_input_checker(next_state,state_size=self.state_size)

        experience = self.experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
 
    def sample(self,batch_size):
        """Randomly sample a batch of experiences from memory.

        Args:
            batch_size (int): size of each training batch
        """
        experiences = random.sample(self.buffer, k=batch_size)

        states = np.stack([e.state.cpu() for e in experiences if e is not None])
        states = torch.from_numpy(np.vstack([e.state.cpu() for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state.cpu() for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    """Deep Q network, simple feedforward neural network.

        Simple Deep Q Network (DQN) algorithm for benchmarking. Follows this tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            num_layers (int): Number of hidden layers
            num_hidden_units (int): Number of units in each hidden layer
            seed (int): Random seed

        Warning:
            This implementation works though the StableBaselines3 implementation is likely better optimized.
    """

    def __init__(self, state_size, action_size, num_layers, num_hidden_units, seed): 
        """Initialize parameters and build model.

        """
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        self.activation = nn.ReLU()

        arch = []
        for i in range(num_layers):
            if i == 0:
                arch.append(nn.Linear(state_size,num_hidden_units))
                arch.append(self.activation)
            elif i == num_layers - 1:
                arch.append(nn.Linear(num_hidden_units,action_size))
            else:
                arch.append(nn.Linear(num_hidden_units,num_hidden_units))
                arch.append(self.activation)

        self.layers = nn.Sequential(*arch)

    def forward(self, state):
        return self.layers(state)
    
    def type_checker(self, x):
        if isinstance(x, dict) and 'state' in x:
            return x['state']
        else:
            return x
    

class DQNAgent(base.Agent):

    """Simple Deep Q Network (DQN) algorithm for benchmarking

        This implementation is based on the PyTorch tutorial found at https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            model (DQN, optional): Predefined model architecture. Defaults to None.
            model_path (str, optional): Path to pretrained model weights. Defaults to None.
            buffer_size (int, optional): replay buffer size. Defaults to int(1e5).
            batch_size (int, optional): minibatch size. Defaults to 64.
            gamma (float, optional): discount factor. Defaults to 0.99.
            lr (float, optional): learning rate. Defaults to 0.001.
            update_every (int, optional): how often to update the network. Defaults to 4.
            do_update (bool, optional): Whether to perform gradient updates during environment interaction. Defaults to False.
            
    """
    def __init__(
        self,
        state_size, 
        action_size, 
        seed,
        model=None,
        model_path=None,
        buffer_size=int(1e5),
        batch_size=64,
        gamma=0.99,
        lr=0.001,
        update_every=4,
        do_update=False,
        total_steps=int(1e6)
    ) -> None:
        
        """Initialize a DQNAgent object.

        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if model:
            self.q_network_local = model
            self.q_network_local = self.q_network_local.to(self.device)
            self.q_network_target = model
            self.q_network_target = self.q_network_target.to(self.device)
        else:
            self.q_network_local = DQN(state_size, action_size, num_layers=3, num_hidden_units=64, seed=seed)
            self.q_network_target = DQN(state_size, action_size, num_layers=3, num_hidden_units=64, seed=seed)

        if model_path:
            model_weights = torch.load(model_path,map_location=self.device,weights_only=True)
            self.q_network_local.load_state_dict(model_weights)
            self.q_network_target.load_state_dict(model_weights)

        self.optimizer = optim.AdamW(self.q_network_local.parameters(), lr=lr,weight_decay=1e-5)
        self.scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps//update_every)

        self.memory = ReplayBuffer(buffer_size, state_size, action_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_every = update_every
        self.t_step = 0
    

        self.do_update = do_update
        self.q_network_local = self.q_network_local.to(self.device)
        self.q_network_target = self.q_network_target.to(self.device)


    def step(self, state, action, reward, next_state, done):
        """Add experience to memory and potentially learn.

        Args:
            state (Union[int, np.ndarray, dict]): current state
            action (int): action taken
            reward (float): reward received
            next_state (Union[int, np.ndarray, dict]): next state
            done (bool): whether the episode has ended
        """
        l = 0
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > 500 * self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                l = self.learn(experiences, self.gamma)
        return l
    def search(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Args:
            state (Union[int, np.ndarray, dict]): current state
            eps (float, optional): epsilon, for epsilon-greedy action selection. Defaults to 0.0
        """
        state = network_input_checker(state,state_size=self.state_size)
        self.q_network_local.eval()
        with torch.no_grad():
            s = network_input_checker(state, self.state_size)
            action_values = self.q_network_local(s)
        self.q_network_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()),action_values.cpu().data.numpy().ravel()
        else:
            return random.choice(np.arange(self.action_size)),action_values.cpu().data.numpy()
    
    def get_guidance(self, state, noise_level = 0):
        """
            Returns the probabilities of selecting
            each available action at given state.
        """
        state = network_input_checker(state,state_size=self.state_size)
        self.q_network_local.eval()
        with torch.no_grad():
            s = network_input_checker(state, self.state_size)
            action_values = self.q_network_local(s).to(torch.float32)
        self.q_network_local.train()
        if noise_level > 0:
            action_values += torch.max(action_values)* torch.normal(0, noise_level, size=action_values.size()).to(self.device)

        
        action_values = action_values.detach().cpu().numpy()
        action_values = action_values - np.max(action_values)
        action_values = np.exp(action_values)
        
        return action_values/np.sum(action_values)  

    def act(self, state,eps=0.0):
        """Returns actions for given state as per current policy.

        Args:
            state (Union[int, np.ndarray, dict]): current state
            eps (float, optional): epsilon, for epsilon-greedy action selection. Defaults to 0.0
        """
        best_action, _ = self.search(state,eps=eps)
        return best_action
    

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        next_states = network_input_checker(next_states,state_size=self.state_size)
        states = network_input_checker(states,state_size=self.state_size)

        Q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)

        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.q_network_local(states).gather(1,actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()


        self.soft_update(self.q_network_local, self.q_network_target, tau=1e-3)
        return loss.detach().cpu().item()
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.

        Args:
            local_model (nn.Module): weights will be copied from
            target_model (nn.Module): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
             target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def epsilon_scheduler(self, aggresive_reduction, slow_reduction, remaining_steps):
        steps1 = aggresive_reduction   # steps from 1 → 0.2
        steps2 = slow_reduction   # steps from 0.2 → 0.01
        steps3 = remaining_steps   # steps remaining at 0.01

        self.epsilon = np.concatenate([
            np.linspace(1, 0.2, steps1, endpoint=False),   # 1 → 0.2 (excludes 0.2)
            np.linspace(0.2, 0.01, steps2, endpoint=False), # 0.2 → 0.01 (excludes 0.01)
            np.full(steps3, 0.01)                           # stays at 0.01
        ])

    def get_epsilon(self, step):
        if hasattr(self, 'epsilon'):
            return self.epsilon[step] if step < self.epsilon.size else self.epsilon[-1]
        else:
            return 0.01  # default epsilon if scheduler not set

def do_gradient_updates(state,env, agent, time_budget,eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Perform gradient updates for a fixed time budget.

    Args:
        state (Union[int, np.ndarray, dict]): current state
        env (gym.Env): environment to interact with
        agent (DQNAgent): DQN agent
        time_budget (float): time budget in seconds
        eps_start (float, optional): starting value of epsilon. Defaults to 1.0.
        eps_end (float, optional): minimum value of epsilon. Defaults to 0.01.
        eps_decay (float, optional): multiplicative factor (per episode) for decreasing epsilon. Defaults to 0.995.
    """
    start = time.time()
    eps = eps_start
    while time.time() - start < time_budget and agent.do_update:
        while True:
            action, values = agent.act(state, eps=0)
            next_state, reward, done, truncated,_ = env.step(action)
            next_state, reward = observation_type_checker(next_state), reward_type_checker(reward)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done or truncated or time.time() - start > time_budget:
                state,_ = env.reset(seed = random.randint(0,100000))

                if isinstance(state, dict) and 'state' in state:
                    state = state['state']

                break
        eps = max(eps_end, eps_decay * eps)






def train_ddqn(env, agent:DQNAgent, training_steps=int(10e6), max_t=200, use_wandb = False):
    """DDQN Training Loop

    Args:
        env (gym.Env): environment to interact with
        agent (DDQN): 
        n_episodes (int, optional): maximum number of training episodes. Defaults to 1000.
        max_t (int, optional): maximum number of timesteps per episode. Defaults to 200.
        eps_start (float, optional): starting value of epsilon, for epsilon-greedy action selection. Defaults to 1.0.
        eps_end (float, optional): minimum value of epsilon. Defaults to 0.01.
        eps_decay (float, optional): multiplicative factor (per episode) for decreasing epsilon. Defaults to 0.999.
    """

    env = RecordVideo(
        env,
        video_folder=f"videos/{env.unwrapped.spec.id}",
        name_prefix="training",
        episode_trigger=lambda x: x % 1_000 == 0  
    )
    scores = []
    scores_window = deque(maxlen=100)
    best_score = -math.inf
    episodic_reward = 0
    episodic_loss = 0

    agent.epsilon_scheduler(
        aggresive_reduction = int(training_steps*0.25),
        slow_reduction = int(training_steps*0.25),
        remaining_steps = int(training_steps*0.5)
    )

    if use_wandb:
        import wandb
        log_frequency = training_steps//100
        wandb.init(project="DQN_Benchmarking", name=f"DQN_{env.spec.id}")
        wandb.config.update({
            "max_t": max_t,
            'environment': env.unwrapped.spec.id
        })
    eps = agent.get_epsilon(0)
    state, _ = env.reset()
    timestep_within_epsiode = 0
    episodes_without_logging = 1
    score = 0
    episodes = 0
    for t in tqdm(range(training_steps)):

        if isinstance(state, dict) and 'state' in state:
            state = state['state']

        if use_wandb and (t % log_frequency == 0):
            wandb.log({
                "episodic reward": episodic_reward/episodes_without_logging,
                "episodic loss": episodic_loss/episodes_without_logging,
                "epsilon": eps,
                "mean_timesteps_per_episode":timestep_within_epsiode/episodes_without_logging,
                'learning rate': agent.optimizer.param_groups[0]['lr']
            })
            episodic_reward = 0
            episodic_loss = 0
            episodes_without_logging = 0
            timestep_within_epsiode = 0

        timestep_within_epsiode += 1
        action, values = agent.search(state, eps)
        next_state, reward, done, truncated,_ = env.step(action)
        episodic_reward += reward
        next_state, reward = observation_type_checker(next_state), reward_type_checker(reward)
        episodic_loss += agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done or truncated:
            episodes_without_logging += 1
            episodes += 1
            scores_window.append(score)
            score = 0
            state,_ = env.reset()

            if  np.mean(scores_window) > best_score and (episodes % 1000 ) == 0:
                best_score = np.mean(scores_window)
                current_dir = os.path.dirname(__file__) 
                os.makedirs(os.path.join(current_dir,f'DDQN_models/{env.unwrapped.spec.id}'), exist_ok=True)
                saved_model_dir = os.path.join(current_dir,f'DDQN_models/{env.unwrapped.spec.id}')
                if t > training_steps//4:
                    torch.save(agent.q_network_local.state_dict(), os.path.join(saved_model_dir,f"DDQN_episode_{t}.pth"))

        eps = agent.get_epsilon(t)


    wandb.finish()
    return agent, scores