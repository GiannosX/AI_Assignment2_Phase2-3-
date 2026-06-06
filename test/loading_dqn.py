'''
In this file, a pretrained dqn agent is loaded and tested on a non-stationary version of the CartPole environment. 
'''
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from agents.dqn import DQNAgent
import  cfgs.cfg_dqn as cfg
import gymnasium as gym
import ns_gym
from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
from ns_gym.update_functions import RandomWalk, IncrementUpdate
import numpy as np


def load_agent(path: str, agent_parameters: dict, env: gym.Env) -> DQNAgent:
    '''
    Loads a pretrained DQN agent from the specified path.
    
    Args:
        path (str): The file path to the saved model.
    '''    
    agent_parameters['state_size'] = env.observation_space.shape[0]
    agent_parameters['action_size'] = env.action_space.n
    agent_parameters['model_path'] = path
    return DQNAgent(**agent_parameters)


def main():
    
    domain = 'CartPole-v1'
    #domain = 'MountainCar-v0'
    env = gym.make(domain, render_mode = 'human')
    #############
    name, version = domain.split("-")

    ########## Step 3: to describe the evolution of the non-stationary parameters, 
    # we define the two schedulers and update functions that model the semi-Markov chain over the relevant parameters
    ############
    scheduler_1 = ContinuousScheduler()
    scheduler_2 = PeriodicScheduler(period=3)

    update_function1= IncrementUpdate(scheduler_1, k=1)
    update_function2 = RandomWalk(scheduler_2)
    if name == "CartPole":
        ##### Step 4: map parameters to update functions
        tunable_params = {"masspole":update_function1, "gravity": update_function2}
    elif name == "MountainCar":
        tunable_params = {"gravity":update_function1, "force": update_function1}

    ######## Step 5: set notification level and pass environment and parameters into wrapper
    ns_env = NSClassicControlWrapper(env,tunable_params,change_notification=True)

    ######### Step 6: set up ns-environment and agent interaction loop. i.e ... 
    done = False
    truncated = False

    episode_reward = 0

    obs,info = ns_env.reset()

    planning_env = ns_env.get_planning_env()
    obs, info = env.reset()
    #agent = load_agent(f"agents/DDQN_models/{domain}/DDQN_episode_2833669.pth", cfg.agent, env)
    #agent = load_agent(f"agents/DDQN_models/{domain}/DDQN_episode_2118775.pth", cfg.agent, env)
    #agent = load_agent(f"agents/DDQN_models/{domain}/DDQN_episode_8502425.pth", cfg.agent, env)
    agent = load_agent(f"agents/DDQN_models/{domain}/DDQN_episode_13271959.pth", cfg.agent, env)
    done = False
    truncated = False
    timestep = 0

    while not (done or truncated):
        #calculate the probabilities
        qs = agent.get_guidance(obs, noise_level = .2)
        #select one action
        action = np.argmax(qs)  # Select the action with the highest Q-value
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        env.render()
        timestep += 1

    print("Episode Reward: ", episode_reward)
    
if __name__ == '__main__':
    main()