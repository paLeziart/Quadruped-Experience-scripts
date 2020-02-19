# coding: utf8

import numpy as np
import matplotlib.pyplot as plt
import foot_trajectory_generator
from time import sleep
import settings


def update_footholds_local(r_feet, S, p_footholds):
    """Checks if foothold positions (local frame) need to be updated. A foothold is updated when a leg ends its swing phase and
    touches the ground at the start of the stance phase.

    Keyword arguments:
    qu -- position/orientation of the robot (6 by 1)
    r_feet -- (x, y) position of the foothold in the base frame (2 by 4)
    S -- foothold sequence of size N by 4 with N the number of step over the prediction horizon
    p_footholds -- the previous foothold positions in wolrd frame (2 by 4)
    """

    #update = ((S[-1, :] == 0)) & (S[0, :] == 1)
    update = (S[0, :] == 1)
    for i in range(4):
        if update[0, i]:
            p_footholds[:, i:(i+1)] = r_feet[:, i:(i+1)]

    return p_footholds


def update_footholds_local_target(r_feet, S, p_footholds):
    """Checks if foothold positions (local frame) need to be updated. A foothold is updated when a leg ends its swing phase and
    touches the ground at the start of the stance phase.

    Keyword arguments:
    qu -- position/orientation of the robot (6 by 1)
    r_feet -- (x, y) position of the foothold in the base frame (2 by 4)
    S -- foothold sequence of size N by 4 with N the number of step over the prediction horizon
    p_footholds -- the previous foothold positions in wolrd frame (2 by 4)
    """

    # update = np.logical_or((S[0, :] == 0),(((S[-1, :] == 0)) & (S[0, :] == 1)))
    update = S[0, :] == 0
    for i in range(4):
        if update[0, i]:
            p_footholds[:, i:(i+1)] = r_feet[:, i:(i+1)]

    return p_footholds


def getFeetTrajectory(feet_start, feet_end, S, dt, T, feet_target_prev, goal_on_ground_prev, ftgs):

    x0 = 0.0
    dx0 = 0.0
    ddx0 = 0.0

    y0 = 0.0
    dy0 = 0.0
    ddy0 = 0.0

    z0 = 0.0
    dz0 = 0.0
    ddz0 = 0.0

    t1 = T

    feet_target = np.zeros((3, 4))

    goal_on_ground = np.zeros((2, 4))

    for i in range(4):
        index = (np.where(S[:, i] == True))[0][0]
        t0 = T - index * dt
        x0 = feet_start[0, i]
        y0 = feet_start[1, i]
        x1 = feet_end[0, i]
        y1 = feet_end[1, i]
        if (t0 != t1):
            [x0, dx0, ddx0,  y0, dy0, ddy0,  z0, dz0, ddz0, gx1, gy1] = (
                ftgs[i]).get_next_foot(x0, dx0, ddx0, y0, dy0, ddy0, x1, y1, t0,  t1, dt)
            feet_target[:, i] = np.array([x0, y0, z0])
            goal_on_ground[:, i] = np.array([gx1, gy1])
        else:
            # np.array([x1, y1, 0.0])
            feet_target[:, i] = feet_target_prev[:, i]
            goal_on_ground[:, i] = goal_on_ground_prev[:, i]

    return feet_target, goal_on_ground, ftgs


# IK for just the 2 links (taken from https://github.com/RationalAsh/invkin/blob/master/invkin.py)
def invkin2(x, y, angleMode=1):
    """Returns the angles of the first two links
    in the robotic arm as a list.
    returns -> (th1, th2)
    input:
    x - The x coordinate of the effector
    y - The y coordinate of the effector
    angleMode - tells the function to give the angle in
                degrees/radians. Default is radians
    output:
    th1 - angle of the first link w.r.t ground
    th2 - angle of the second link w.r.t the first"""

    # Stuff for calculating th2
    r_2 = x**2 + y**2
    l_sq = l1**2 + l2**2
    term2 = (r_2 - l_sq)/(2*l1*l2)
    term1 = ((1 - term2**2)**0.5)*-1
    # Calculate th2
    th2 = math.atan2(term1, term2)
    # Optional line. Comment this one out if you
    # notice any problems
    th2 = -1*th2

    # Stuff for calculating th2
    k1 = l1 + l2*math.cos(th2)
    k2 = l2*math.sin(th2)
    r = (k1**2 + k2**2)**0.5
    gamma = math.atan2(k2, k1)
    # Calculate th1
    th1 = math.atan2(y, x) - gamma

    if(angleMode == 1):
        return th1, th2
    else:
        return math.degrees(th1), math.degrees(th2)


def update_target_footholds_no_lock(v_ref, q, v, t_stance, S, dt, T, k=0.03, h=0.25, g=9.81):
    """Returns a 2 by 4 matrix containing the [x, y]^T position of the next desired footholds for the four feet
    For feet in a swing phase it is where they should land and for feet currently touching the ground it is
    where they should land at the end of their next swing phase

    Keyword arguments:
    v_ref -- reference velocity vector of the flying base (6 by 1, linear and angular stacked)
    q -- current position/orientation of the flying base (6 by 1)
    v -- current velocity vector of the flying base (6 by 1, linear and angular stacked)
    """

    # Order of feet: FL, FR, HL, HR

    # Start with shoulder term
    p = np.tile(np.array([[0], [0]]), (1, 4)) + settings.shoulders  # + np.dot(R, shoulders)

    # Shift initial position of contact outwards for more stability
    p[1, :] += np.array([0.025, -0.025, 0.025, -0.025])

    # Add symmetry term
    p += t_stance * 0.5 * v[0:2, 0:1]

    # Add feedback term
    p += k * (v[0:2, 0:1] - v_ref[0:2, 0:1])

    # Add centrifugal term
    cross = np.cross(v[0:3, 0:1], v_ref[3:6, 0:1], 0, 0).T
    p += 0.5 * np.sqrt(h/g) * cross[0:2, 0:1]

    # Time remaining before the end of the currrent swing phase
    t_remaining = np.zeros((1, 4))
    for i in range(4):
        indexes_stance = (np.where(S[:, i] == True))[0]
        indexes_swing = (np.where(S[:, i] == False))[0]
        # index = (np.where(S[:, i] == True))[0][0]
        if (S[0, i] == True) and (S[-1, i] == False):
            t_remaining[0, i] = T
        else:
            index = (indexes_stance[indexes_stance > indexes_swing[0]])[0]
            t_remaining[0, i] = index * dt

    # Add velocity forecast
    #  p += np.tile(v[0:2, 0:1], (1, 4)) * t_remaining
    for i in range(4):
        yaw = np.linspace(0, t_remaining[0, i]-dt, np.floor(t_remaining[0, i]/dt)) * v[5, 0]
        p[0, i] += (dt * np.cumsum(v[0, 0] * np.cos(yaw) - v[1, 0] * np.sin(yaw)))[-1]
        p[1, i] += (dt * np.cumsum(v[0, 0] * np.sin(yaw) + v[1, 0] * np.cos(yaw)))[-1]

    return p


def update_desired_feet_pos(feet_current, feet_end, S, dt, T, current_feet_pos, ftgs, desired_vel_feet, desired_acc_feet, desired_pos_feet, target_footholds_with_lock, desired_pos_feet_w, target_footholds_with_lock_w, q_w):

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
        t0 = T - index * dt

        # Current position of the foot
        x0 = feet_current[0, i]
        y0 = feet_current[1, i]

        # Target position of the foot
        x1 = feet_end[0, i]
        y1 = feet_end[1, i]

        # Update if the foot is in swing phase or is going to leave the ground
        if ((S[0, i] == True) and (S[1, i] == False)):
            t0 = 0
        if (t0 != t1):

            # Get desired 3D position
            [x0, dx0, ddx0,  y0, dy0, ddy0,  z0, dz0, ddz0, gx1, gy1] = (ftgs[i]).get_next_foot(
                x0, desired_vel_feet[0, i], desired_acc_feet[0, i], y0, desired_vel_feet[1, i], desired_acc_feet[1, i], x1, y1, t0,  t1, dt)

            # Retrieve result in terms of position, velocity and acceleration
            desired_pos_feet[:, i] = np.array([x0, y0, z0])
            desired_vel_feet[:, i] = np.array([dx0, dy0, dz0])
            desired_acc_feet[:, i] = np.array([ddx0, ddy0, ddz0])
            desired_pos_feet_w[:, i:(i+1)] = np.vstack((q_w[0:2, 0:1], np.zeros((1, 1)))) + \
                np.dot(R, desired_pos_feet[:, i:(i+1)])

            # Update target position of the foot with lock
            target_footholds_with_lock[:, i] = np.array([gx1, gy1])
            target_footholds_with_lock_w[:, i:(i+1)] = q_w[0:2, 0:1] + \
                np.dot(R[0:2, 0:2], target_footholds_with_lock[:, i:(i+1)])

    return desired_pos_feet, target_footholds_with_lock, ftgs, desired_vel_feet, desired_acc_feet, desired_pos_feet_w, target_footholds_with_lock_w
