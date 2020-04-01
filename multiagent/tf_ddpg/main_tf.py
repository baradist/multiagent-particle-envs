# from utils import plotLearning
import os

import gym
import numpy as np

from multiagent.tf_ddpg.ddpg_agent import Agent

if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    env_name = 'BipedalWalker-v3'
    env = gym.make(env_name)

    agent = Agent(alpha=0.0001, beta=0.001, input_dims=[24], tau=0.001, env=env, n_actions=4,
                  chkpt_dir='tmp/ddpg/' + env_name)

    np.random.seed(0)

    agent.load_models()

    score_history = []
    while agent.count < 5000:
        agent.count += 1
        obs = env.reset()
        done = False
        score = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
            # env.render()

        score_history.append(score)

        saving_step = 200
        if agent.count % saving_step == 0:
            print('episode ', agent.count, ', mean score %.2f' % np.mean(score_history[-saving_step:]),
                  'training 1000 games avg %.2f' % np.mean(score_history[-1000:]))
            agent.save_models()

        # filename = 'bipedal.png'
        # plotLearning(score_hostory, filename, window=100)
