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
        # for i, agent in enumerate(self.agents):
        #     self.process_ball(agent)
        return super().step(action_n)

    def process_ball(self, agent):
        ball = self.world.ball
        is_close, dist, hypot = close_to_each_other(agent, ball)
        if is_close:
            # move the ball, so that not to intersect
            ball.state.p_pos[0] = agent.state.p_pos[0] \
                                  - (agent.state.p_pos[0] - ball.state.p_pos[0]) * hypot / dist
            ball.state.p_pos[1] = agent.state.p_pos[1] \
                                  - (agent.state.p_pos[1] - ball.state.p_pos[1]) * hypot / dist
            # kick the ball
            agent_abs_vel = math.sqrt(
                agent.state.p_vel[0] * agent.state.p_vel[0] + agent.state.p_vel[1] * agent.state.p_vel[1])
            ball.state.p_vel[0] += (ball.state.p_pos[0] - agent.state.p_pos[0]) * agent_abs_vel
            ball.state.p_vel[1] += (ball.state.p_pos[1] - agent.state.p_pos[1]) * agent_abs_vel

def close_to_each_other(agent, ball):
    x = abs(agent.state.p_pos[0] - ball.state.p_pos[0])
    y = abs(agent.state.p_pos[1] - ball.state.p_pos[1])
    hypot = agent.size + ball.size
    distance = math.sqrt(x * x + y * y)
    return distance < hypot, distance, hypot
