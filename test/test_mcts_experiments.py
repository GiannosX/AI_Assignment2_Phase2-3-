import sys
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from copy import deepcopy

# Add project root to path for imports
project_root = str(Path(__file__).resolve().parents[1])
sys.path.append(project_root)

from agents.MCTS import StandardMCTSAgent, InformedMCTSAgent
from agents.dqn import DQNAgent
import cfgs.cfg_dqn as cfg

from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
from ns_gym.update_functions import RandomWalk, IncrementUpdate

def setup_ns_env(domain, render_mode=None):
    """Initializes the non-stationary environment exactly as in the template."""
    env = gym.make(domain, render_mode=render_mode)
    name, version = domain.split("-")
    
    scheduler_1 = ContinuousScheduler()
    scheduler_2 = PeriodicScheduler(period=3)
    update_function1 = IncrementUpdate(scheduler_1, k=1)
    update_function2 = RandomWalk(scheduler_2)
    
    if name == "CartPole":
        tunable_params = {"masspole": update_function1, "gravity": update_function2}
    elif name == "MountainCar":
        tunable_params = {"gravity": update_function1, "force": update_function1}
    else:
        tunable_params = {}
        
    ns_env = NSClassicControlWrapper(env, tunable_params, change_notification=True)
    return ns_env

def load_dqn_mentor(domain, env):
    """Loads the pretrained DQN agent based on the domain."""
    agent_parameters = deepcopy(cfg.agent)
    agent_parameters['state_size'] = env.observation_space.shape[0]
    agent_parameters['action_size'] = env.action_space.n
    
    # Select the correct model path based on the domain
    if "CartPole" in domain:
        model_path = os.path.join(project_root, "agents", "DDQN_models", "CartPole-v1", "DDQN_episode_13271959.pth")
    else:
        model_path = os.path.join(project_root, "agents", "DDQN_models", "MountainCar-v0", "DDQN_episode_2118775.pth")
    
    agent_parameters['model_path'] = model_path
    return DQNAgent(**agent_parameters)

def run_experiment(domain, num_episodes=500, num_repetitions=3):
    """Runs the full experimental procedure for a given domain."""
    print(f"\n--- Starting experiments for {domain} ---")
    
    # Data structure to store rewards for each repetition
    # Format: { algorithm_name: [ [rep1_rewards], [rep2_rewards], [rep3_rewards] ] }
    all_results = {
        "Standard MCTS": [],
        "Informed MCTS (Noise 0.0)": [],
        "Informed MCTS (Noise 0.1)": [],
        "Informed MCTS (Noise 0.3)": []
    }
    
    for rep in range(num_repetitions):
        print(f"\nRepetition {rep+1}/{num_repetitions}")
        
        # Setup env and mentor for this repetition
        ns_env = setup_ns_env(domain, render_mode=None)
        mentor = load_dqn_mentor(domain, ns_env)
        
        # Initialize agents
        # Note: planning_env is passed but will be deepcopied inside the act method
        agents_to_test = [
            ("Standard MCTS", StandardMCTSAgent(ns_env.get_planning_env())),
            ("Informed MCTS (Noise 0.0)", InformedMCTSAgent(ns_env.get_planning_env(), mentor, noise_level=0.0)),
            ("Informed MCTS (Noise 0.1)", InformedMCTSAgent(ns_env.get_planning_env(), mentor, noise_level=0.1)),
            ("Informed MCTS (Noise 0.3)", InformedMCTSAgent(ns_env.get_planning_env(), mentor, noise_level=0.3))
        ]
        
        for alg_name, agent in agents_to_test:
            print(f"Running {alg_name}...")
            episodic_rewards = []
            
            for ep in tqdm(range(num_episodes), desc=f"Episodes"):
                obs, info = ns_env.reset()
                done = False
                truncated = False
                episode_reward = 0
                
                while not (done or truncated):
                    planning_env = ns_env.get_planning_env()
                    action = agent.act(obs, planning_env)
                    obs, reward, done, truncated, info = ns_env.step(action)
                    episode_reward += reward
                
                episodic_rewards.append(episode_reward)
            
            all_results[alg_name].append(episodic_rewards)
                
    return all_results

def plot_and_save(domain, results):
    """Calculates means and plots the results."""
    plt.figure(figsize=(12, 7))
    
    for alg_name, repetitions in results.items():
        # Convert to numpy array for easy calculation
        data = np.array(repetitions)
        mean_rewards = np.mean(data, axis=0)
        
        # Plot the mean rewards
        plt.plot(mean_rewards, label=alg_name)
        
        # Optional: Add shaded area for variance (min/max across repetitions)
        min_rewards = np.min(data, axis=0)
        max_rewards = np.max(data, axis=0)
        plt.fill_between(range(len(mean_rewards)), min_rewards, max_rewards, alpha=0.1)
        
    plt.title(f"MCTS Performance in Non-Stationary {domain}")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Episodic Reward")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    output_path = f"{domain}_mcts_comparison.png"
    plt.savefig(output_path)
    print(f"\nSaved comparison plot to {output_path}")
    plt.close()

if __name__ == "__main__":
    # Check for test mode (shorter run)
    is_test = True # Set to False for the full assignment run
    
    if is_test:
        print("RUNNING IN TEST MODE (Reduced episodes and repetitions)")
        num_episodes = 5
        num_repetitions = 1
    else:
        num_episodes = 500
        num_repetitions = 3
        
    domains = ['CartPole-v1', 'MountainCar-v0']
    
    for domain in domains:
        results = run_experiment(domain, num_episodes=num_episodes, num_repetitions=num_repetitions)
        plot_and_save(domain, results)
        
    print("\nExperiments completed successfully.")
