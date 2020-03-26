import math

import numpy as np

from multiagent.environment import MultiAgentEnv


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class FootballEnvironment(MultiAgentEnv):
    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):
        super(FootballEnvironment, self).__init__(world, reset_callback, reward_callback,
                                                  observation_callback, info_callback,
                                                  done_callback, shared_viewer)

    def step(self, action_n):
            obs_n = []
            reward_n = []
            done_n = []
            info_n = {'n': []}
            self.agents = self.world.policy_agents
            # set action for each agent
            for i, agent in enumerate(self.agents):
                self._set_action(action_n[i], agent, self.action_space[i])
            # advance world state
            self.world.step()
            for i, agent in enumerate(self.agents):
                self.process_ball(agent)
            # record observation for each agent
            for agent in self.agents:
                obs_n.append(self._get_obs(agent))
                reward_n.append(self._get_reward(agent))
                done_n.append(self._get_done(agent))

                info_n['n'].append(self._get_info(agent))

            # all agents get total reward in cooperative case
            reward = np.sum(reward_n)
            if self.shared_reward:
                reward_n = [reward] * self.n

            return obs_n, reward_n, done_n, info_n

    def process_ball(self, agent):
        ball = self.world.landmarks[0]
        is_close, dist, hypot = close_to_each_other(agent, ball)
        if is_close:
            # move the ball, so that not to intersect
            ball.state.p_pos[0] = agent.state.p_pos[0] \
                                  - (agent.state.p_pos[0] - ball.state.p_pos[0]) * hypot / dist
            ball.state.p_pos[1] = agent.state.p_pos[1] \
                                  - (agent.state.p_pos[1] - ball.state.p_pos[1]) * hypot / dist
            # kick the ball
            agent_abs_vel = ball.max_speed * math.sqrt(
                agent.state.p_vel[0] * agent.state.p_vel[0] + agent.state.p_vel[1] * agent.state.p_vel[1])
            ball.state.p_vel[0] += (ball.state.p_pos[0] - agent.state.p_pos[0]) * agent_abs_vel
            ball.state.p_vel[1] += (ball.state.p_pos[1] - agent.state.p_pos[1]) * agent_abs_vel

def close_to_each_other(agent, ball):
    x = abs(agent.state.p_pos[0] - ball.state.p_pos[0])
    y = abs(agent.state.p_pos[1] - ball.state.p_pos[1])
    hypot = agent.size + ball.size
    distance = math.sqrt(x * x + y * y)
    return distance < hypot, distance, hypot
