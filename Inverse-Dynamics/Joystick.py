# coding: utf8

import numpy as np
from time import clock


class Joystick:
    """Joystick-like controller that outputs the reference velocity in local frame
    """

    def __init__(self):

        # Starting time if we want to ouput reference velocities based on elapsed time
        self.t_start = clock()

        # Reference velocity in local frame
        self.v_ref = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

    def update_v_ref(self, k_loop):

        # Change reference velocity during the simulation (in trunk frame)
        """if k == 51:
            settings.v_ref = np.array([[0.2, 0, 0.0, 0, 0, 0.0]]).T
        if k == 101:
            settings.v_ref = np.array([[0.2, 0, 0.0, 0, 0, 0.4]]).T
        if k == 201:
            settings.v_ref = np.array([[0.2, 0, 0.0, 0, 0, 0.0]]).T
        if k == 251:
            settings.v_ref = np.array([[0.2, 0, 0.0, 0, 0, -0.4]]).T
        if k == 351:
            settings.v_ref = np.array([[0.2, 0, 0.0, 0, 0, 0.0]]).T
        if k == 401:
            settings.v_ref = np.array([[-0.0, -0.2, 0.0, 0, 0, 0.0]]).T
        if k == 401:
            settings.v_ref = np.array([[-0.2, 0, 0.0, 0, 0, 0.0]]).T
        if k == 501:
            settings.v_ref = np.array([[-0.0, -0.2, 0.0, 0, 0, 0.0]]).T
        if k == 601:
            settings.v_ref = np.array([[-0.0, -0.0, 0.0, 0, 0, 0.4]]).T
        if k == 701:
            settings.v_ref = np.array([[-0.0, -0.2, -0.0, 0, 0, -0.4]]).T"""

        if k_loop == 151:
            self.v_ref = np.array([[0.1, 0, 0.0, 0, 0, 0.4]]).T
        if k_loop == 301:
            self.v_ref = np.array([[0.1, 0, 0.0, 0, 0, -0.4]]).T
        if k_loop == 451:
            self.v_ref = np.array([[0.0, -0.1, 0.0, 0, 0, 0.0]]).T
        if k_loop == 600:
            self.v_ref = np.array([[-0.1, 0.1, 0.0, 0, 0, -0.4]]).T

        return 0
