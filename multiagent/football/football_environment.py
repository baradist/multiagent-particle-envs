import math

import numpy as np

from multiagent.environment import MultiAgentEnv


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
from multiagent.football.gate import Gate


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

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            # print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                if issubclass(type(entity), Gate): # TODO: Rectangle instead of Gate
                    geom = rendering.make_polygon(entity.v)
                else:
                    geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results
