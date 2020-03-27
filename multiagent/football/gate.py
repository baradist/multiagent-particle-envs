import numpy as np
from multiagent.core import Entity


class Gate(Entity):
    def __init__(self):
        super(Gate, self).__init__()
        self.width = 1
        self.height = .1
        w_half = self.width / 2
        h_half = self.height / 2
        self.v = [[-w_half, -h_half], [-w_half, h_half], [w_half, h_half], [w_half, -h_half]]

        self.collide = False

        self.color = np.array([0.75, 0.75, 0.25])
        self.state.p_vel = np.array([0., 0.])
