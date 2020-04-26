import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.football.football_environment import FootballEnvironment
from multiagent.football.gate import Gate
from multiagent.scenario import BaseScenario


def goal(ball, gate):
    ball_pos = ball.state.p_pos
    gate_pos = gate.state.p_pos
    gate_half_width = gate.width / 2
    gate_half_height = gate.height / 2
    return gate_pos[0] - gate_half_width < ball_pos[0] < gate_pos[0] + gate_half_width \
           and gate_pos[1] - gate_half_height < ball_pos[1] < gate_pos[1] + gate_half_height


class Scenario(BaseScenario):
    def get_env(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):
        return FootballEnvironment(world, reset_callback, reward_callback,
                 observation_callback, info_callback,
                 done_callback, shared_viewer)

    def make_world(self):
        world = World()
        # add agents
        num_agents = 2
        num_adversaries = 1
        world.num_agents = num_agents

        # world.collaborative = True

        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.1
            agent.initial_mass = 5
            agent.max_speed = 5
        for i in range(world.num_agents):
            world.agents[i].color = np.array([0.85, 0.35, 0.35]) if world.agents[i].adversary \
                else np.array([0.35, 0.35, 0.85])

        ball = Landmark()
        ball.name = 'ball'
        ball.size = .1
        ball.max_speed = 30
        ball.collide = True
        ball.movable = True
        ball.initial_mass = .2
        world.landmarks.append(ball)
        self.ball = ball

        self.gate_adv = Gate()
        self.gate_adv.state.p_pos = np.array([0., 1.])
        world.landmarks.append(self.gate_adv)
        self.gate_agent = Gate()
        self.gate_agent.state.p_pos = np.array([0., -1.])
        world.landmarks.append(self.gate_agent)
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for landmarks
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.color = np.array([0.75,0.75,0.75])
        world.landmarks[0].color = np.array([0.75,0.25,0.25]) # self.ball
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        self.ball.state.p_pos = np.array([0., 0.])
        self.ball.state.p_vel = np.array([0., 0.])
        self.is_done = False

    def reward(self, agent, world):
        return self.agent_reward(agent, self.gate_agent, self.gate_adv) if agent.adversary \
            else self.agent_reward(agent, self.gate_adv, self.gate_agent)

    def agent_reward(self, agent, gate, our_gate):
        # goal
        if goal(self.ball, gate):
            self.is_done = True
            return 10000
        if goal(self.ball, our_gate):
            self.is_done = True
            return -10000
        ball_pos = self.ball.state.p_pos
        if any(ball_pos < [-1., -1.]) or any(ball_pos > [1., 1.]):
            self.is_done = True
            # return -500 # TODO treat the agent who kicked out
        gate_pos = gate.state.p_pos
        ball_gate_sq_dist = np.sum(np.square(ball_pos - gate_pos))

        # agent-ball
        ball_agent_sq_dist = np.sum(np.square(ball_pos - agent.state.p_pos))
        is_close = np.square(self.ball.size + agent.size) >= ball_agent_sq_dist
        if is_close:
            return 10

        return -5 * ball_gate_sq_dist - ball_agent_sq_dist

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            # if not entity.boundary:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        is_self_adversary = 1. if agent.adversary else 0.
        # communication of all other agents
        other_pos = []
        other_vel = []
        other_is_adv = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            # if not other.adversary:
            other_vel.append(other.state.p_vel)
            other_is_adv.append(1. if other.adversary else 0.)
            result = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
        return np.append(np.append(result, other_is_adv), is_self_adversary)
#np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel) + [is_self_adversary]
    def done(self, agent, world):
        return self.is_done
