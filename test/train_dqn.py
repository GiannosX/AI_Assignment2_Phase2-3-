import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from agents.dqn import DQNAgent, train_ddqn
import gymnasium as gym
import cfgs.cfg_dqn as cfg

def create_dqn(path:str, state_size: int, action_size:int, total_steps:int) ->DQNAgent:
    """ 
     Methof that creates a DQN agent. The agent's configuration should exist in
     a path+'/cfg.py' file
    """
    agent_params = cfg.agent | {'state_size' : state_size, 'action_size' : action_size, 'total_steps' : total_steps}

    return DQNAgent(**agent_params)


def create_env(domain: str, render_mode: str = None) -> gym.Env:


    # Initialise the environment
    env = gym.make(domain, render_mode=render_mode)
    # Reset the environment to generate the first observation
    observation, info = env.reset(seed=42)
    return env


def main():
    domain = 'CartPole-v1'
    #domain = 'MountainCar-v0'
    env = create_env(domain, render_mode='rgb_array')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    horizon = int(15e6) if domain == 'CartPole-v1' else int(5e6)
    agent = create_dqn(
        path = '../cfgs/cfgs_dqn.py',
        state_size = state_size,
        action_size = action_size,
        total_steps = horizon
    )

    obs, _ = env.reset()
    print('Checking whether DQN works or not')
    for _ in range(1_000):
        action = agent.act(obs, eps = 0)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    print('DQN is working properly')

    print('Training DQN agent')
    train_ddqn(
        env = env,
        agent = agent,
        training_steps = horizon,
        max_t = 1_000,
        use_wandb = True,
    )

if __name__ == '__main__':
    main()

