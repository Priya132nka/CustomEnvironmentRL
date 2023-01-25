from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack
import os
import gym
from gym.spaces import Discrete, Box
import random
import pygame
import numpy as np


class GridBallEnv(gym.Env):
    def __init__(self, size=6):
        self.size = size
        self.render_mode = "human"
        self.window_size = 600
        self.action_space = Discrete(4)
        self.action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([-1, 0]),  # left
            2: np.array([0, 1]),  # up
            3: np.array([0, -1])  # down
        }
        self.observation_space = Box(0, size-1, shape=(2,), dtype=int)
        self.agent_state = None
        self.danger_state1 = None
        self.danger_state2 = None
        self.target_state = None
        self.window = None
        self.clock = None

    def step(self, action):
        direction = self.action_to_direction[int(action)]
        self.agent_state = np.clip(self.agent_state + direction * 100, 0, (self.size-1) * 100)
        done = np.array_equal(self.agent_state, self.target_state)
        if done:
            reward = 1
        elif np.array_equal(self.agent_state, self.danger_state1) or np.array_equal(self.agent_state, self.danger_state2):
            reward = -1
        else:
            reward = 0
        observation = self.agent_state
        info = {}
        return observation, reward, done, info

    def render(self, mode="human"):
        if self.render_mode == "human":
            if self.clock is None:
                self.clock = pygame.time.Clock()
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
            background = pygame.Surface((self.window_size, self.window_size))
            background.fill((255, 255, 255))
            square_size = (self.window_size / self.size)
            pygame.draw.rect(background, (0, 255, 0), pygame.Rect(self.target_state, (100, 100)))
            pygame.draw.rect(background, (255, 0, 0), pygame.Rect(self.danger_state1, (100, 100)))
            pygame.draw.rect(background, (255, 0, 0), pygame.Rect(self.danger_state2, (100, 100)))

            for x in range(0, self.size + 1):
                pygame.draw.line(background, (0, 0, 0), (0, x * 100), (self.window_size, x * 100), width=2)
                pygame.draw.line(background, (0, 0, 0), (x * 100, 0), (x * 100, self.window_size), width=2)
            pygame.draw.circle(background, (0, 0, 255), self.agent_state + 50, 33)
            self.window.blit(background, background.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(4)

    def reset(self):

        self.agent_state = np.array([0, 0])
        self.target_state = np.array([500, 500])
        self.danger_state1 = np.array([200, 400])
        self.danger_state2 = np.array([400, 300])
        observation = self.agent_state
        return observation


if __name__ == "__main__":
    env = GridBallEnv()
    log_path = os.path.join('Training', 'Logs')
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=100000)
    model.save('PPO')
    env.render("human")
    # evaluate_policy(model, env, n_eval_episodes=5, render=True)
    # print("Entering")
    # obs = env.reset()

    episodes = 6
    for episode in range(1, episodes):
        obs = env.reset()
        score = 0
        done = False
        while not done:
            env.render("human")
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)

            score += reward
        print('Episode:{} Score:{}'.format(episode, score))
    env.close()
