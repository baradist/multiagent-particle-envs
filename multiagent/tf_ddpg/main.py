import os
import time

import numpy as np

from multiagent import scenarios
from multiagent.tf_ddpg.ddpg_agent import Agent


def make_env(scenario_name):
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = scenario.get_env(world, scenario.reset_world, scenario.reward, scenario.observation,
                           done_callback=scenario.done)
    return env


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # variables
    env_name = 'simple'
    env = make_env(env_name)
    max_episode_len = 60
    saving_step = 500
    num_episodes = 20000
    display = True

    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    agent = Agent(alpha=0.0001, beta=0.001, input_dims=[4], tau=0.001, env=env, n_actions=5,
                  chkpt_dir='tmp/ddpg/' + env_name)
    trainers = [agent]

    np.random.seed(0)

    agent.load_models()

    score_history = []
    while agent.count < num_episodes or display:
        agent.count += 1
        obs_n = env.reset()
        done = False
        score = 0
        episode_step = 0
        while not (done or (episode_step >= max_episode_len)):
            episode_step += 1
            action_n = [agent.choose_action(obs) for agent, obs in zip(trainers, obs_n)]
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            done = all(done_n)
            for i, agent in enumerate(trainers):
                agent.remember(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], int(done_n[i]))
            obs_n = new_obs_n

            agent.learn()
            if display:
                env.render()
                time.sleep(0.05)

            score += np.mean(rew_n)

        score_history.append(score)

        if agent.count % saving_step == 0:
            print('episode ', agent.count, ', mean score %.2f' % np.mean(score_history[-saving_step:]),
                  'training 1000 games avg %.2f' % np.mean(score_history[-1000:]))
            agent.save_models()
