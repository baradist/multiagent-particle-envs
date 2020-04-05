from pathlib import Path

import numpy as np
import tensorflow as tf
from pyglet.resource import file

from multiagent.ddpg.action_noise import OUActionNoise
from multiagent.ddpg.replay_buffer import ReplayBuffer
from multiagent.tf_ddpg.actor import Actor
from multiagent.tf_ddpg.critic import Critic


class Agent(object):
    def __init__(self, name, alpha, beta, input_dims, tau, env, gamma=0.99, n_actions=2, max_size=1000000,
                 layer1_size=400, layer2_size=300, batch_size=64, chkpt_dir='tmp/ddpg', action_bound=None):
        self.name = name
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.sess = tf.Session()

        Path(chkpt_dir).mkdir(parents=True, exist_ok=True)
        self.chkpt_dir = chkpt_dir

        self.actor = Actor(alpha, n_actions, self.name + 'Actor', input_dims, self.sess, layer1_size, layer2_size,
                           action_bound, chkpt_dir=chkpt_dir)
        self.critic = Critic(beta, 5, self.name + 'Critic', [14], self.sess, layer1_size, layer2_size,
                             chkpt_dir=chkpt_dir)

        self.target_actor = Actor(alpha, n_actions, self.name + 'TargetActor', input_dims, self.sess, layer1_size, layer2_size,
                                  action_bound, chkpt_dir=chkpt_dir)
        self.target_critic = Critic(beta, 5, self.name + 'TargetCritic', [14], self.sess, layer1_size, layer2_size,
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

    def learn(self, trainers):
        if self.memory.mem_cntr < self.batch_size:
            return
        # Sample a random minibatch of N transitions (si, ai, ri, si+1) from R
        sample_index = self.memory.make_index(self.batch_size)
        obs_n = []
        obs_next_n = []
        act_n = []
        for i, t in enumerate(trainers):
            obs, act, rew, obs_next, done = t.memory.sample_buffer(sample_index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        target_act_next_n = [t.target_actor.predict(obs_next_n[i]) for i, t in enumerate(trainers)]
        # target_act_next = self.target_actor.predict(new_state)
        critic_value_ = self.target_critic.predict(obs_next_n, target_act_next_n)  # TODO flatten

        # Set yi = ri + γQ0(si+1, µ0(si+1|θµ0)|θQ0) )
        target_q = []
        for j in range(self.batch_size):
            target_q.append(reward[j] + self.gamma * critic_value_[j] * done[j])  # done = 1 - (done from the env)
        target_q = np.reshape(target_q, (self.batch_size, 1))
        # train q network
        # Update critic by minimizing the loss: L = 1 N P i (yi − Q(si , ai |θ Q))2
        _ = self.critic.train(obs_n, act_n, target_q)
        # train p network
        # Update the actor policy using the sampled policy gradient:
        # ∇θµ J ≈ 1 / N X i ∇aQ(s, a|θ Q)|s=si,a=µ(si)∇θµ µ(s|θ µ )|si
        a_outs = self.actor.predict(state)
        grads = self.critic.get_action_gradients(state, a_outs)
        self.actor.train(state, grads[0])
        # Update the target networks:
        # θ Q0 ← τθQ + (1 − τ )θQ
        # 0 θ µ 0 ← τθµ + (1 − τ )θ µ 0
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
