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
                           done_callback=scenario.done, discrete_action_space=False)
    return env


def get_trainers(env, num_adversaries=0):  # TODO: num_adversaries
    trainers = []
    for i in range(num_adversaries):
        trainers.append(Agent("agent_%d" % i,
                              alpha=0.0001, beta=0.001, input_dims=env.observation_space[i].shape, tau=0.001, env=env,
                              n_actions=env.action_space[i].shape[0],
                              chkpt_dir='tmp/ddpg/' + env_name + '/' + "agent_%d" % i,
                              action_bound=env.action_space[i].high))
    for i in range(num_adversaries, env.n):
        trainers.append(Agent("agent_%d" % i,
                              alpha=0.0001, beta=0.001, input_dims=env.observation_space[i].shape, tau=0.001, env=env,
                              n_actions=env.action_space[i].shape[0],
                              chkpt_dir='tmp/ddpg/' + env_name + '/' + "agent_%d" % i,
                              action_bound=env.action_space[i].high))
    for _, trainer in enumerate(trainers):
        trainer.load_models()

    return trainers


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # variables
    env_name = 'simple_speaker_listener'
    env = make_env(env_name)
    max_episode_len = 50
    saving_step = 100
    num_episodes = 60000
    display = False
    # display = True

    np.random.seed(0)

    trainers = get_trainers(env)

    score_history = []
    agent = trainers[0]  # TODO
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
            for i, trainer in enumerate(trainers):
                trainer.remember(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], int(done_n[i]))
            obs_n = new_obs_n

            if display:
                env.render()
                time.sleep(0.05)

            score += np.mean(rew_n)

        for _, trainer in enumerate(trainers):  # TODO: how often should we learn?
            trainer.learn(trainers)

        score_history.append(score)

        if agent.count % saving_step == 0:
            print()
            print('episode ', agent.count, ', mean score %.2f' % np.mean(score_history[-saving_step:]),
                  'training 1000 games avg %.2f' % np.mean(score_history[-1000:]))
            for _, trainer in enumerate(trainers):
                trainer.save_models()

        if agent.count % (saving_step / 10) == 0:
            print(' ' + str(agent.count) + ' ', end='')
        if agent.count % (saving_step / 100) == 0:
            print('.', end='')
