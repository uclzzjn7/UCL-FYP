import os

import numpy as np
import pybullet as p
from env import ClutteredPushGrasp
from robot import UR5Robotiq85
from utilities import YCBModels, Camera


def run_simulation():
    ycb_models = YCBModels(os.path.join('./data/ycb', '**', 'textured-decmp.obj'))  # Load YCB models
    camera = Camera((1, 1, 1), (0, 0, 0), (0, 0, 1), 0.1, 5, (320, 320), 40)        # Initialize camera
    robot = UR5Robotiq85((0, 0.5, 0), (0, 0, 0))                                    # Initialize robot

    # Initialize simulation environment
    env = ClutteredPushGrasp(robot, ycb_models, camera, vis=True)
    env.reset()
    env.container.resetObject()

    while True:
        env.step(env.read_debug_parameter(), 'end')
        env.digit_step()


if __name__ == '__main__':
    run_simulation()
