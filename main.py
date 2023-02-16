import gym
import time
import gym_taco_environments
from agent import MonteCarlo
import argparse


def train(env, agent, episodes):
    for _ in range(episodes):
        observation, _ = env.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = agent.get_action(observation)
            new_observation, reward, terminated, truncated, _ = env.step(action)
            agent.update(observation, action, reward, terminated)
            observation = new_observation


def play(env, agent):
    observation, _ = env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated):
        action = agent.get_best_action(observation)
        observation, _, terminated, truncated, _ = env.step(action)
        env.render()
        time.sleep(1)

def define_parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--height", help="Matrix height", type=int)
    parser.add_argument("-c", "--width", help="Matrix width", type=int)
    parser.add_argument("-p", "--problem", help="Problem ID", type=str)
    parser.add_argument("-f", "--file", help="Filepath", type=str)
    parser.add_argument("-s", "--seed", help="Seed", type=int)

    return parser.parse_args()

if __name__ == "__main__":
    arguments = define_parse_arguments()
    env = gym.make("FrozenMaze-v0", render_mode="human", delay=0.5)
    agent = MonteCarlo(
        env.observation_space.n, env.action_space.n, gamma=0.9, epsilon=0.9
    )

    train(env, agent, episodes=3000000)
    agent.render()

    play(env, agent)

    env.close()