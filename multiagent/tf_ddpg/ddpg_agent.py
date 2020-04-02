from pathlib import Path

import numpy as np
import tensorflow as tf
from pyglet.resource import file

from multiagent.ddpg.action_noise import OUActionNoise
from multiagent.ddpg.replay_buffer import ReplayBuffer
from multiagent.tf_ddpg.actor import Actor
from multiagent.tf_ddpg.critic import Critic


class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, n_actions=2, max_size=1000000,
                 layer1_size=400, layer2_size=300, batch_size=64, chkpt_dir='tmp/ddpg'):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.sess = tf.Session()

        Path(chkpt_dir).mkdir(parents=True, exist_ok=True)
        self.chkpt_dir = chkpt_dir

        self.actor = Actor(alpha, n_actions, 'Actor', input_dims, self.sess, layer1_size, layer2_size,
                           # env.action_space.high, chkpt_dir=chkpt_dir)
                           [1., 1., 1., 1., 1.], chkpt_dir=chkpt_dir)
        self.critic = Critic(beta, n_actions, 'Critic', input_dims, self.sess, layer1_size, layer2_size,
                             chkpt_dir=chkpt_dir)

        self.target_actor = Actor(alpha, n_actions, 'TargetActor', input_dims, self.sess, layer1_size, layer2_size,
                                  # env.action_space.high, chkpt_dir=chkpt_dir)
                                  [1., 1., 1., 1., 1.], chkpt_dir=chkpt_dir)
        self.target_critic = Critic(beta, n_actions, 'TargetCritic', input_dims, self.sess, layer1_size, layer2_size,
                                    chkpt_dir=chkpt_dir)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_critic = \
            [self.target_critic.params[i].assign(
                tf.multiply(self.critic.params[i], self.tau) \
                + tf.multiply(self.target_critic.params[i], 1. - self.tau))
                for i in range(len(self.target_critic.params))]

        self.update_actor = \
            [self.target_actor.params[i].assign(
                tf.multiply(self.actor.params[i], self.tau) \
                + tf.multiply(self.target_actor.params[i], 1. - self.tau))
                for i in range(len(self.target_actor.params))]

        self.sess.run(tf.global_variables_initializer())
        self.update_network_parameters(first=True)
        self.count = 0

    def update_network_parameters(self, first=False):
        if first:
            old_tau = self.tau
            self.tau = 1.0
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)
            self.tau = old_tau
        else:
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        mu = self.actor.predict(state)  # returns list of lists

        mu_prime = mu + self.noise()

        return mu_prime[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        critic_value_ = self.target_critic.predict(new_state, self.target_actor.predict(new_state))

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma * critic_value_[j] * done[j])  # done = 1 - (done from the env)
        target = np.reshape(target, (self.batch_size, 1))

        _ = self.critic.train(state, action, target)

        a_outs = self.actor.predict(state)
        grads = self.critic.get_action_gradients(state, a_outs)
        self.actor.train(state, grads[0])

        self.update_network_parameters()

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
        self.save_count()

    def load_models(self):
        try:
            self.actor.load_checkpoint()
            self.target_actor.load_checkpoint()
            self.critic.load_checkpoint()
            self.target_critic.load_checkpoint()
            self.load_count()
        except Exception as e:
            self.count = 0

    def save_count(self):
        # print('... saving counter ...')
        f = open(self.chkpt_dir + '/count', 'w')
        f.write(str(self.count))
        f.close()

    def load_count(self):
        print('... loading counter ...')
        f = file(self.chkpt_dir + '/count', 'r')
        self.count = int(f.read())
        f.close()
