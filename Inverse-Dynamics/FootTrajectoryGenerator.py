# coding: utf8

import numpy as np
import foot_trajectory_generator as ftg


class FootTrajectoryGenerator:
    """A foot trajectory generator that handles the generation of a 3D trajectory
    with a 5th order polynomial to lead each foot from its location at the start of
    its swing phase to its final location that has been decided by the FootstepPlanner

    :param shoulders: A 2 by 4 numpy array, the position of shoulders in local frame
    :param dt: A float, time step of the contact sequence
    """

    def __init__(self, shoulders, dt):

        # Position of shoulders in local frame
        self.shoulders = shoulders

        # Time step of the trajectory generator
        self.dt = dt

        # Desired (x, y) position of footsteps without lock mechanism before impact
        # Received from the FootstepPlanner
        # self.footsteps = self.shoulders.copy()

        # Desired (x, y) position of footsteps with lock mechanism before impact
        self.footsteps_lock = self.shoulders.copy()

        # Desired footsteps with lock in world frame for visualisation purpose
        self.footsteps_lock_world = self.footsteps_lock.copy()

        # Desired position, velocity and acceleration of feet in 3D, in local frame
        self.desired_pos = np.vstack((shoulders, np.zeros((1, 4))))
        self.desired_vel = np.zeros(self.desired_pos.shape)
        self.desired_acc = np.zeros(self.desired_pos.shape)

        # Desired 3D position in world frame for visualisation purpose
        self.desired_pos_world = self.desired_pos.copy()

        # Maximum height at which the robot should lift its feet during swing phase
        self.max_height_feet = 0.03

        # Lock target positions of footholds before touchdown
        self.t_lock_before_touchdown = 0.01

        # Foot trajectory generator objects (one for each foot)
        self.ftgs = [ftg.Foot_trajectory_generator(
            self.max_height_feet, self.t_lock_before_touchdown) for i in range(4)]

        # Initialization of ftgs objects
        for i in range(4):
            self.ftgs[i].x1 = self.desired_pos[0, i]
            self.ftgs[i].y1 = self.desired_pos[1, i]

        self.flag_initialisation = False

    def update_desired_feet_pos(self, footsteps_target, S, S_dt, T, q_w):

        # Initialisation of rotation from local frame to world frame
        c, s = np.cos(q_w[5, 0]), np.sin(q_w[5, 0])
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        # Initialisation of trajectory parameters
        x0 = 0.0
        dx0 = 0.0
        ddx0 = 0.0

        y0 = 0.0
        dy0 = 0.0
        ddy0 = 0.0

        z0 = 0.0
        dz0 = 0.0
        ddz0 = 0.0

        # The swing phase lasts T seconds
        t1 = T

        # For each foot
        for i in range(4):
            # Time remaining before touchdown
            index = (np.where(S[:, i] == True))[0][0]
            t0 = T - index * S_dt

            # Current position of the foot
            x0 = self.desired_pos[0, i]
            y0 = self.desired_pos[1, i]

            # Target position of the foot
            x1 = footsteps_target[0, i]
            y1 = footsteps_target[1, i]

            # Update if the foot is in swing phase or is going to leave the ground
            if ((S[0, i] == True) and (S[1, i] == False)):
                t0 = 0

            if (t0 != t1) and (t0 != (t1 - S_dt)):

                # Get desired 3D position
                [x0, dx0, ddx0,  y0, dy0, ddy0,  z0, dz0, ddz0, gx1, gy1] = (self.ftgs[i]).get_next_foot(
                    x0, self.desired_vel[0, i], self.desired_acc[0, i],
                    y0, self.desired_vel[1, i], self.desired_acc[1, i],
                    x1, y1, t0,  t1, self.dt)

                if self.flag_initialisation:
                    # Retrieve result in terms of position, velocity and acceleration
                    self.desired_pos[:, i] = np.array([x0, y0, z0])
                    self.desired_vel[:, i] = np.array([dx0, dy0, dz0])
                    self.desired_acc[:, i] = np.array([ddx0, ddy0, ddz0])

                    # Update target position of the foot with lock
                    self.footsteps_lock[:, i] = np.array([gx1, gy1])

                    # Update variables in world frame
                    self.desired_pos_world[:, i:(i+1)] = np.vstack((q_w[0:2, 0:1], np.zeros((1, 1)))) + \
                        np.dot(R, self.desired_pos[:, i:(i+1)])
                    self.footsteps_lock_world[:, i:(i+1)] = q_w[0:2, 0:1] + \
                        np.dot(R[0:2, 0:2], self.footsteps_lock[:, i:(i+1)])
            else:
                self.desired_vel[:, i] = np.array([0.0, 0.0, 0.0])
                self.desired_acc[:, i] = np.array([0.0, 0.0, 0.0])

        if not self.flag_initialisation:
            self.flag_initialisation = True

        return 0

    def update_frame(self, vel):
        """As we are working in local frame, the footsteps drift backwards
        if the trunk is moving forwards as footsteps are not supposed to move
        in the world frame

        Keyword arguments:
        vel -- Current velocity vector of the flying base (6 by 1, linear and angular stacked)
        """

        # Displacement along x and y
        c, s = np.cos(- vel[5, 0] * self.dt), np.sin(- vel[5, 0] * self.dt)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        # Update desired 3D position
        self.desired_pos = np.dot(R, self.desired_pos) - \
            self.dt * np.vstack((np.tile(vel[0:2, 0:1], (1, 4)), np.zeros((1, 4))))

        # Update desired 2D location of footsteps
        self.footsteps_lock = np.dot(R[0:2, 0:2], self.footsteps_lock) \
            - self.dt * np.tile(vel[0:2, 0:1], (1, 4))

        return 0

    def update_viewer(self, viewer, initialisation):
        """Update display for visualization purpose

        Keyword arguments:
        :param viewer: A gepetto viewer object
        :param initialisation: A bool, is it the first iteration of the main loop
        """

        # Display locked target footholds with red spheres (gepetto gui)
        rgbt = [1.0, 0.0, 0.0, 0.5]
        for i in range(4):
            if initialisation:
                viewer.gui.addSphere("world/sphere"+str(i)+"_lock", .025, rgbt)  # .1 is the radius
            viewer.gui.applyConfiguration("world/sphere"+str(i)+"_lock",
                                          (self.footsteps_lock_world[0, i], self.footsteps_lock_world[1, i],
                                           0.0, 1., 0., 0., 0.))

        # Display desired 3D position of feet with magenta spheres (gepetto gui)
        rgbt = [1.0, 0.0, 1.0, 0.5]
        for i in range(4):
            if initialisation:
                viewer.gui.addSphere("world/sphere"+str(i)+"_des", .03, rgbt)  # .1 is the radius
            viewer.gui.applyConfiguration("world/sphere"+str(i)+"_des",
                                          (self.desired_pos_world[0, i], self.desired_pos_world[1, i],
                                           self.desired_pos_world[2, i], 1., 0., 0., 0.))

        return 0
