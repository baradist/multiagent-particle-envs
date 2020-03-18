import os

from multiagent.ddpg.ddpg_agent import Agent
import gym
import numpy as np
import time
# from utils import plotLearning

env = gym.make(('LunarLanderContinuous-v2'))

agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
              batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)


def train():
    # agent.load_models()
    np.random.seed(0)
    score_history = []
    while (agent.count < 1000):
        done = False
        score = 0
        obs = env.reset()
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
            env.render()

        agent.count += 1
        score_history.append(score)
        print('episode', agent.count, 'score %.2f' % score, '100 game average %.2f' % np.mean(score_history[-100:]))
        if agent.count % 25 == 0:
            agent.save_models()

        filename = 'lunar-lander.png'
        # plotLearning(score_history, filename, window=100)

def play():
    agent.load_models()
    np.random.seed(0)
    score_history = []
    for i in range(1000):
        done = False
        score = 0
        obs = env.reset()
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            score += reward
            obs = new_state
            env.render()
            time.sleep(0.01)

        score_history.append(score)
        print('episode', i, 'score %.2f' % score, '100 game average %.2f' % np.mean(score_history[-100:]))


train()
