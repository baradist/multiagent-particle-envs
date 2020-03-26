import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.football.football_environment import FootballEnvironment
from multiagent.football.gate import Gate
from multiagent.scenario import BaseScenario

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
            agent.collide = False
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.05
        for i in range(world.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85]) if world.agents[i].adversary \
                else np.array([0.85, 0.35, 0.35])

        ball = Landmark()
        ball.name = 'ball'
        ball.size = .02
        ball.max_speed = 30
        ball.collide = False
        ball.movable = True
        world.landmarks.append(ball)
        self.ball = ball

        self.gate_adv = Gate()
        self.gate_adv.state.p_pos = np.array([0, 1])
        world.landmarks.append(self.gate_adv)
        self.gate_agent = Gate()
        self.gate_agent.state.p_pos = np.array([0, -1])
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
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
        #     landmark.state.p_vel = np.zeros(world.dim_p)
        self.ball.state.p_pos = np.array([0., 0.])
        self.ball.state.p_vel = np.array([0., 0.])

    def reward(self, agent, world):
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # goal
        if self.ball.state.p_pos[1] > .97 and self.ball.state.p_pos[0] > -.25 and self.ball.state.p_pos[0] < .25:
            return 10000

        gate_pos = [0, 1]
        ball_gate_sq_dist = np.sum(np.square(self.ball.state.p_pos - gate_pos))

        # agent-ball
        ball_agent_sq_dist = np.sum(np.square(self.ball.state.p_pos - agent.state.p_pos))
        is_close = np.square(self.ball.size + agent.size) >= ball_agent_sq_dist
        if is_close:
            return 10

        return -5 * ball_gate_sq_dist - ball_agent_sq_dist

    def adversary_reward(self, agent, world):
        if self.ball.state.p_pos[1] < -.97 and self.ball.state.p_pos[0] > -.25 and self.ball.state.p_pos[0] < .25:
            return 10000

        gate_pos = [0, -1]
        ball_gate_sq_dist = np.sum(np.square(self.ball.state.p_pos - gate_pos))

        # agent-ball
        ball_agent_sq_dist = np.sum(np.square(self.ball.state.p_pos - agent.state.p_pos))
        is_close = np.square(self.ball.size + agent.size) >= ball_agent_sq_dist
        if is_close:
            return 10

        return -5 * ball_gate_sq_dist - ball_agent_sq_dist

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)
