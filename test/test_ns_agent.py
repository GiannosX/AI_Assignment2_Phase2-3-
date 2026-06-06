###### Step 1: Import necessary gym and ns_gym modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from agents.MCTS import RandomAgent

import gymnasium as gym
import ns_gym
from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
from ns_gym.update_functions import RandomWalk, IncrementUpdate
from tqdm import tqdm


if __name__ == '__main__':
    ###### Step 2: Create a standard gym environment ####
    domain = 'CartPole-v1'
    domain = 'MountainCar-v0'
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

    
    done = False
    truncated = False
    obs,info = ns_env.reset()
    planning_env = ns_env.get_planning_env()
    agent = RandomAgent(planning_env)

    timestep = 0
    for episodes in tqdm(range(10)):
        obs,info = ns_env.reset()
        planning_env = ns_env.get_planning_env()
        episode_reward = 0
        done = False
        truncated = False
        while not (done or truncated):
            action = agent.act(obs,planning_env)
            #print(f"timestep: {timestep}, action: {action}")
            obs, reward, done, truncated, info = ns_env.step(action)
            env.render()
            planning_env = ns_env.get_planning_env()
            episode_reward += reward
            timestep += 1
            if (done or truncated):
                break

