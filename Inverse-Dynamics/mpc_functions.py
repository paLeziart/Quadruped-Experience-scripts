# coding: utf8

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_slsqp
import scipy.sparse

from time import sleep, clock

import settings
from feet_functions import update_target_footholds_no_lock


def getRotMatrix(rpy):
    c, s = np.cos(rpy[0, 0]), np.sin(rpy[0, 0])
    R1 = np.array(((1, 0, 0), (0, c, -s), (0, s, c)))

    c, s = np.cos(rpy[1, 0]), np.sin(rpy[1, 0])
    R2 = np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))

    c, s = np.cos(rpy[2, 0]), np.sin(rpy[2, 0])
    R3 = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))

    return np.dot(R1, np.dot(R2, R3))


def getSkew(a):
    """Returns the skew matrix of a 3 by 1 column vector

    Keyword arguments:
    a -- the column vector
    """
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=a.dtype)


def footStepPlanner(p_prev, v_ref, qu_m, vu_m, t_stance, S, dt, T, k=0.03, h=0.25, g=9.81):
    """Returns a 2 by 4 matrix containing the [x, y]^T position of the next desired footholds for the four feet
    For feet in a swing phase it is where they should land and for feet currently touching the ground it is
    where they should land at the end of their next swing phase

    Keyword arguments:
    v_ref -- reference velocity vector of the flying base (6 by 1, linear and angular stacked)
    qu_m -- current position/orientation of the flying base (6 by 1)
    vu_m -- current velocity vector of the flying base (6 by 1, linear and angular stacked)
    """

    t_remaining = np.zeros((1, 4))
    for i in range(4):
        index = (np.where(S[:, i] == True))[0][0]
        t_remaining[0, i] = index * dt

    #print("v_ref: ", v_ref.T)
    #print("vu_m: ", vu_m.T)

    c, s = np.cos(-qu_m[5, 0]), np.sin(-qu_m[5, 0])
    R = np.array([[c, -s, 0., 0., 0., 0.], [s, c, 0., 0., 0., 0], [0., 0., 1.0, 0., 0., 0.],
                  [0., 0., 0., c, -s, 0.], [0., 0., 0., s, c, 0.], [0., 0., 0., 0., 0., 1.0]])
    vu_m_rot = np.dot(R, vu_m.copy())

    # FL, FR, HL, HR

    # p = np.tile(qu_m[0:2, 0:1], (1, 4)) + np.dot(R, settings.shoulders)
    p = np.tile(np.array([[0], [0]]), (1, 4)) + settings.shoulders  # + np.dot(R, shoulders)
    # print(p.shape)
    p += t_stance * 0.5 * vu_m_rot[0:2, 0:1]
    p += k * (vu_m_rot[0:2, 0:1] - v_ref[0:2, 0:1])
    cross = np.cross(vu_m_rot[0:3, 0:1], v_ref[3:6, 0:1], 0, 0).T
    p += 0.5 * np.sqrt(h/g) * cross[0:2, 0:1]

    for i in range(4):
        c, s = np.cos(vu_m_rot[5, 0] * t_remaining[0, i]), np.sin(vu_m_rot[5, 0] * t_remaining[0, i])
        R = np.array([[c, -s], [s, c]])
        p[0:2, i:(i+1)] = np.dot(R, p[0:2, i:(i+1)])

    # Go from trunk frame to global frame
    c, s = np.cos(qu_m[5, 0]), np.sin(qu_m[5, 0])
    R = np.array([[c, -s], [s, c]])
    p = np.dot(R, p)
    p += np.tile(np.array([[qu_m[0, 0]], [qu_m[1, 0]]]), (1, 4))

    # Add velocity forecast
    p += np.tile(vu_m[0:2, 0:1], (1, 4)) * t_remaining

    # Update only during swing phase
    for i in range(4):
        if (S[0, i] == 1):
            p[:, i:(i+1)] = p_prev[:, i:(i+1)].copy()

    '''print("p:         ", p)
    print("v_ref:     ", v_ref.T)
    print("qu_m:      ", qu_m.T)
    print("vu_m:      ", vu_m.T)
    print("shoulders: ", settings.shoulders)'''
    return p


def footSequence(t, dt, T_gait, phases):
    """Returns the sequence of footholds from time t to time t+T_gait with step dt and a phase offset.
    The output is a matrix of size N by 4 with N the number of time steps (around T_gait / dt). Each column
    corresponds to one foot with 1 if it touches the ground or 0 otherwise.

    Keyword arguments:
    t -- current time
    dt -- time step
    T_gait -- period of the current gait
    phases -- phase offset for each foot compared to the default sequence
    """

    t_seq = np.matrix(np.linspace(t, t+T_gait-dt, int(np.round(T_gait/dt)))).T
    phases_seq = (np.hstack((t_seq, t_seq, t_seq, t_seq)) - phases * T_gait) * 2 * np.pi / T_gait
    S_feet = (np.sin(phases_seq) >= 0).astype(float)

    # To have a four feet stance phase at the start
    # Likely due to numerical effect we don't have it
    S_feet[0, :] = np.ones((1, 4))
    S_feet[int(S_feet.shape[0]*0.5), :] = np.ones((1, 4))

    if np.any(np.sum(S_feet, axis=1) > 3):
        print("boop")

    return S_feet


def low_pass_robot(qu, vu):
    """Mimics the behaviour of a real robot to test the MPC

    Keyword arguments:
    qu -- body position/orientation command from the MPC
    vu -- body linear/angular velocity command from the MPC
    """

    # qu = qu + vu * settings.dt

    return qu, vu


def getRobotStatesDuringTrajectory(x, f, dt, S, n_contacts, footholds):
    """Returns the future trajectory of the robot for each time step of the
    predition horizon. The ouput is a matrix of size 12 by N with N the number
    of time steps (around T_gait / dt) and 12 the position / orientation /
    linear velocity / angular velocity vertically stacked.

    Keyword arguments:
    x -- current position/orientation of the robot (12 by 1)
    f -- ground reaction forces for each step and for each foot in contact
    dt -- time step
    """

    # global S, p_contacts, t

    c, s = np.cos(x[5, 0]), np.sin(x[5, 0])
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    A = np.zeros((12, 12))
    A[0:3, 0:3] = np.eye(3)
    A[0:3, 6:9] = dt * np.eye(3)

    A[3:6, 3:6] = np.eye(3)
    A[3:6, 9:12] = dt * R

    A[6:9, 6:9] = np.eye(3)
    A[9:12, 9:12] = np.eye(3)

    gI = np.diag([0.00578574, 0.01938108, 0.02476124])
    #gI = np.eye(3)
    gI_inv = np.linalg.inv(gI)
    m = 2.2

    x_mpc = np.zeros((12, len(f)))
    g = np.zeros((12, 1))
    g[8, 0] = -9.81*dt

    f_extern = np.zeros((12, 1))
    '''if (settings.t > 0.3) and (settings.t < 0.4):
        f_extern[7, 0] = +100
        print("Force")'''

    for i in range(len(f)):
        nb_contacts = n_contacts[i, 0]
        B = np.zeros((12, 3 * nb_contacts))
        pos_contacts = footholds[:, (S[i, :] == 1).getA()[0, :]]
        for j in range(nb_contacts):
            '''contact_foot = np.array([[pos_contacts[0, j]], [pos_contacts[1, j]], [0]]) - np.array([[x[0, 0]], [x[1, 0]], [0]]) 
            B[6:9, (3*j):(3*j+3)] = dt * np.dot(gI_inv, getSkew(contact_foot))
            B[9:12, (3*j):(3*j+3)] = dt / m * np.eye(3)'''

        if i == 0:
            for j in range(nb_contacts):
                contact_foot = np.array([[pos_contacts[0, j]], [pos_contacts[1, j]], [0]]) - x[0:3, 0:1]
                B[9:12, (3*j):(3*j+3)] = dt * np.dot(gI_inv, getSkew(contact_foot))
                B[6:9, (3*j):(3*j+3)] = dt / m * np.eye(3)
            x_mpc[:, i:(i+1)] = np.dot(A, x) + np.dot(B, f[i]) + g + f_extern
        else:
            # print("###")
            # print(f[i].shape)
            if f[i].shape[0] == 0:
                print(f[i])
            for j in range(nb_contacts):
                contact_foot = np.array([[pos_contacts[0, j]], [pos_contacts[1, j]], [0]]) - x_mpc[0:3, (i-1):i]
                B[9:12, (3*j):(3*j+3)] = dt * np.dot(gI_inv, getSkew(contact_foot))
                B[6:9, (3*j):(3*j+3)] = dt / m * np.eye(3)
            x_mpc[:, i:(i+1)] = np.dot(A, x_mpc[:, (i-1):i]) + np.dot(B, f[i]) + g + f_extern

    return x_mpc


def getRobotStatesDuringTrajectoryBis(x, f, dt, S, n_contacts, footholds):
    """Returns the future trajectory of the robot for each time step of the
    predition horizon. The ouput is a matrix of size 12 by N with N the number
    of time steps (around T_gait / dt) and 12 the position / orientation /
    linear velocity / angular velocity vertically stacked.

    Keyword arguments:
    x -- current position/orientation of the robot (12 by 1)
    f -- ground reaction forces for each step and for each foot in contact
    dt -- time step
    """

    # global S, p_contacts, t

    c, s = np.cos(x[5, 0]), np.sin(x[5, 0])
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    A = np.zeros((12, 12))
    A[0:3, 0:3] = np.eye(3)
    A[0:3, 6:9] = dt * np.eye(3)

    A[3:6, 3:6] = np.eye(3)
    A[3:6, 9:12] = dt * R

    A[6:9, 6:9] = np.eye(3)
    A[9:12, 9:12] = np.eye(3)

    gI = np.diag([0.00578574, 0.01938108, 0.02476124])
    # gI = np.eye(3)
    gI_inv = np.linalg.inv(gI)
    m = 2.2

    x_mpc = np.zeros((12, n_contacts.shape[0]))
    g = np.zeros((12, 1))
    g[8, 0] = -9.81*dt

    f_extern = np.zeros((12, 1))
    '''if (settings.t > 0.3) and (settings.t < 0.4):
        f_extern[7, 0] = +100
        print("Force")'''

    n_tot = 0

    for i in range(n_contacts.shape[0]):
        nb_contacts = n_contacts[i, 0]
        B = np.zeros((12, 3 * nb_contacts))
        pos_contacts = footholds[:, (S[i, :] == 1).getA()[0, :]]
        for j in range(nb_contacts):
            '''contact_foot = np.array([[pos_contacts[0, j]], [pos_contacts[1, j]], [0]]) - np.array([[x[0, 0]], [x[1, 0]], [0]]) 
            B[6:9, (3*j):(3*j+3)] = dt * np.dot(gI_inv, getSkew(contact_foot))
            B[9:12, (3*j):(3*j+3)] = dt / m * np.eye(3)'''

        if i == 0:
            for j in range(nb_contacts):
                contact_foot = np.array([[pos_contacts[0, j]], [pos_contacts[1, j]], [0]]) - x[0:3, 0:1]
                B[9:12, (3*j):(3*j+3)] = dt * np.dot(gI_inv, getSkew(contact_foot))
                B[6:9, (3*j):(3*j+3)] = dt / m * np.eye(3)
            x_mpc[:, i:(i+1)] = np.dot(A, x) + np.dot(B, f[(0+n_tot)
                        :(0+n_tot+3*settings.n_contacts[i, 0]), 0:1]) + g + f_extern
        else:
            # print("###")
            # print(f[i].shape)
            if f[i].shape[0] == 0:
                print(f[i])
            for j in range(nb_contacts):
                contact_foot = np.array([[pos_contacts[0, j]], [pos_contacts[1, j]], [0]]) - x_mpc[0:3, (i-1):i]
                B[9:12, (3*j):(3*j+3)] = dt * np.dot(gI_inv, getSkew(contact_foot))
                B[6:9, (3*j):(3*j+3)] = dt / m * np.eye(3)
            x_mpc[:, i:(i+1)] = np.dot(A, x_mpc[:, (i-1):i]) + np.dot(B, f[(0+n_tot)
                        :(0+n_tot+3*settings.n_contacts[i, 0]), 0:1]) + g + f_extern

        n_tot += 3 * settings.n_contacts[i, 0]

    return x_mpc


def getRefStatesDuringTrajectory(qu, v_ref, dt, T_gait, h):
    """Returns the reference trajectory of the robot for each time step of the
    predition horizon. The ouput is a matrix of size 12 by N with N the number
    of time steps (around T_gait / dt) and 12 the position / orientation /
    linear velocity / angular velocity vertically stacked.

    Keyword arguments:
    qu -- current position/orientation of the robot (6 by 1)
    v_ref -- reference velocity vector of the flying base (6 by 1, linear and angular stacked)
    dt -- time step
    T_gait -- period of the current gait
    """

    n_steps = int(np.round(T_gait/dt))
    qu_ref = np.zeros((6, n_steps))

    dt_vector = np.linspace(dt, T_gait, n_steps)
    qu_ref = v_ref * dt_vector

    """yaw = 0
    prev_pos = np.zeros((2,1))
    for i in range(n_steps):
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        yaw += dt * v_ref[5, 0]
        qu_ref[0:2, i:(i+1)] = prev_pos + dt * np.dot(R, v_ref[0:2, 0:1])
        prev_pos = qu_ref[0:2, i:(i+1)]"""
        
    yaw = np.linspace(0, T_gait-dt, n_steps) * v_ref[5, 0]
    qu_ref[0, :] = dt * np.cumsum(v_ref[0, 0] * np.cos(yaw) - v_ref[1, 0] * np.sin(yaw))
    qu_ref[1, :] = dt * np.cumsum(v_ref[0, 0] * np.sin(yaw) + v_ref[1, 0] * np.cos(yaw))

    # Stack the reference velocity to the reference position to get the reference state vector
    x_ref = np.vstack((qu_ref, np.tile(v_ref, (1, n_steps))))

    # Height is supposed constant
    x_ref[2, :] = h

    return x_ref


def getCost(x_mpc, x_ref, f_mpc, Q, R):
    """Returns the cost associated with the MPC cost function

    Keyword arguments:
    x_mpc -- future trajectory of the robot (size 12 by N) with the lumped mass model
    x_ref -- reference trajectory of the robot (size 12 by N)
    f_mpc -- future ground reaction forces for each step (list of N arrays of size k x 3 with k the nb of contacts during that step)
    Q -- cost matrix for the state vector
    R -- cost function for the ground reaction forces
    """

    diff = np.abs(x_mpc - x_ref)
    c = np.sum(np.diag(np.dot(np.dot(diff.T, Q), diff)))

    c_bis = 0
    for i in range(diff.shape[1]):
        vec = diff[:, i:(i+1)]
        c_bis += np.sum((np.dot(np.dot(vec.T, Q), vec)))

    '''for i in range(len(f_mpc)):
        f_step = (f_mpc[i]).reshape((-1, 3))
        c += np.sum(np.diag(np.dot(np.dot(f_step, R), f_step.T)))'''

    return c


def cost(x_f_stacked):
    """Splits the optimization vector to retrieve x and f that will be used
    by other functions.

    Keyword arguments:
    x_f_stacked -- optimization vector of shape K by 1 with x and all f[i] arrays 
    verticaly stacked
    """

    # global settings.n_contacts, dt, xref

    # From array to matrix
    x_f_stacked = np.matrix(x_f_stacked).T

    # Retrieve state vector
    x = x_f_stacked[0:12, 0:1]
    x = np.vstack((settings.qu_m, settings.vu_m))

    # Retrieve force arrays
    '''f = []
    n_tot = 0
    for i in range(settings.n_contacts.shape[0]):
        f_step = x_f_stacked[(0+n_tot):(0+n_tot+3*settings.n_contacts[i, 0]), 0:1]
        #f_step = np.reshape(f_step, (-1, 1))
        f.append(f_step)
        if f[i].shape[0] == 0:
            print(settings.n_contacts)
        n_tot += 3 * settings.n_contacts[i, 0]'''

    # Get the robot trajectory during the prediction horizon
    # x_robot = getRobotStatesDuringTrajectory(x, f, settings.dt, settings.S, settings.n_contacts, settings.footholds)
    x_robot = getRobotStatesDuringTrajectoryBis(
        x, x_f_stacked, settings.dt, settings.S, settings.n_contacts, settings.footholds)

    """plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(x_robot[0,:], x_robot[1,:], x_robot[2,:], "--", linewidth=5, color='r')
    plt.plot(settings.x_ref[0,:], settings.x_ref[1,:], settings.x_ref[2,:], "--", linewidth=3, color='b')
    #plt.show()
    print("###")
    print(x_robot[0:3,:])
    print(settings.x_ref[0:3,:])
    print("###")
    plt.draw()
    plt.pause(0.2)"""

    Q = 1 * np.eye(12)
    for i in range(6):
        Q[i, i] = 5
    R = 0.0 * np.eye(3)

    # return getCost(x_robot, settings.x_ref, f, Q, R)
    return getCost(x_robot, settings.x_ref, x_f_stacked, Q, R)


class my_CallbackLogger:
    def __init__(self):
        self.nfeval = 1

    def __call__(self, x):

        if self.nfeval == 49:
            # From array to matrix
            x_f_stacked = np.matrix(x).T

            # Retrieve state vector
            x = np.vstack((settings.qu_m, settings.vu_m))

            # Get the robot trajectory during the prediction horizon
            x_robot = getRobotStatesDuringTrajectoryBis(
                x, x_f_stacked, settings.dt, settings.S, settings.n_contacts, settings.footholds)

            plt.close("all")
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plt.plot(x_robot[0, :], x_robot[1, :], x_robot[2, :], "--", linewidth=5, color='r')
            plt.plot(settings.x_ref[0, :], settings.x_ref[1, :], settings.x_ref[2, :], "--", linewidth=3, color='b')
            plt.xlim([-0.1, 0.4])
            plt.ylim([-1, 1])
            ax.set_zlim([-0.1, 0.4])
            # plt.show()
            print("###")
            print(x_robot[0:3, :])
            print(settings.x_ref[0:3, :])
            print("###")
            plt.draw()
            plt.pause(0.2)

            Q = 1 * np.eye(12)
            for i in range(6):
                Q[i, i] = 5
            R = 0.0 * np.eye(3)
            c = getCost(x_robot, settings.x_ref, x_f_stacked, Q, R)

            flag = 1
        self.nfeval += 1


def constraint_eq(x_f_stacked):
    """Assesses the value of equality constraints. If a constraint is respected then
    it must return 0 (if 3*x = y is the constraint then we returns the value of 3 * x - y).

    Keyword arguments:
    x_f_stacked -- optimization vector of shape K by 1 with x and all f[i] arrays 
    verticaly stacked
    """

    return np.array([0])


def constraint_ineq(x_f_stacked):
    """Assesses the value of inequality constraints. If a constraint is respected then
    it must return a positive value (if 3*x > y is the constraint then we returns the value of 3 * x - y).

    Keyword arguments:
    x_f_stacked -- optimization vector of shape K by 1 with x and all f[i] arrays 
    verticaly stacked
    """

    nu = 0.7

    # From array to matrix
    x_f_stacked = np.matrix(x_f_stacked).T

    # Retrieve force arrays
    out = np.array([[]])
    n_tot = 0
    for i in range(settings.n_contacts.shape[0]):
        f_step = x_f_stacked[(0+n_tot):(0+n_tot+3*settings.n_contacts[i, 0]), 0:1]
        n_tot += 3 * settings.n_contacts[i, 0]
        f_step = f_step.reshape((-1, 3))

        ieq = nu * f_step[:, 2] - np.abs(f_step[:, 0])
        out = np.hstack((out, ieq.T))

    return np.array([0])  # out.getA()[0, :]


def getOptimizationVector(qu, v, f):
    """Stacks state vector, velocity vector and ground forces list into a single column vector.

    Keyword arguments:
    qu -- position/orientation of the robot (6 by 1)
    v -- velocity vector of the flying base (6 by 1, linear and angular stacked)
    f -- ground reaction forces for each step (list of N arrays of size k x 3 with k the nb of contacts during that step)
    """

    #x_opt = np.vstack((qu, v))
    # for i in range(len(f)):
    #    x_opt = np.vstack((x_opt, np.reshape(f[i], (-1, 1))))
    #print("x_opt shape: ", x_opt.shape)

    x_opt = np.reshape(f[0], (-1, 1))
    for i in range(1, len(f)):
        x_opt = np.vstack((x_opt, np.reshape(f[i], (-1, 1))))

    return x_opt


def getNumContactsVector(f):
    """Returns the total number of contacts over the prediction horizon for each time step.
    If 2 feet touch the ground during 3 steps than 4 feet during 2 steps
    it will returns [2, 2, 2, 4, 4]^T

    Keyword arguments:
    f -- ground reaction forces for each step (list of N arrays of size k x 3 with k the nb of contacts during that step)
    """

    settings.n_contacts = np.zeros((len(f), 1))
    for i in range(len(f)):
        settings.n_contacts[i, 0] = (f[i]).shape[0]
    return settings.n_contacts


def update_contacts(qu, r_feet, future_touchdowns, S, p_contacts):
    """Checks if foothold positions (world frame) need to be updated. A foothold is updated when a leg ends its swing phase and
    touches the ground at the start of the stance phase.

    Keyword arguments:
    qu -- position/orientation of the robot (6 by 1)
    r_feet -- (x, y) position of the foothold in the base frame (2 by 4)
    S -- foothold sequence of size N by 4 with N the number of step over the prediction horizon
    p_contacts -- the previous foothold positions in wolrd frame (2 by 4)
    """
    c, s = np.cos(qu[5, 0]), np.sin(qu[5, 0])
    R = np.array([[c, -s], [s, c]])

    update = ((S[-1, :] == 0)) & (S[0, :] == 1)
    #update = np.logical_or((S[0, :] == 0),(((S[-1, :] == 0)) & (S[0, :] == 1)))

    # print(update)
    for i in range(4):
        if update[0, i]:
            if i == 1:
                flag = 1
            p_contacts[:, i:(i+1)] = r_feet[:, i:(i+1)]
            inter = qu[0:2, 0:1] + np.dot(R, r_feet[0:2, i:(i+1)])
            settings.footholds[0:2, i:(i+1)] = r_feet[:, i:(i+1)]  # inter.copy()

    update = (S[0, :] == 0)
    for i in range(4):
        if update[0, i]:
            settings.footholds[0:2, i:(i+1)] = future_touchdowns[:, i:(i+1)]

    """print("qu: ", qu.T)
    print("contacts: ", p_contacts)"""
    # print(settings.footholds)

    return p_contacts


def initSparseConstraintsMatrices(n_x, n_f, dt, S, n_contacts, footholds, xref, x0):
    t_test = clock()

    # Number of timesteps in the prediction horizon
    nb_xf = n_contacts.shape[0]

    # Mass of the quadruped in [kg] (found in urdf)
    m = 2.2

    # Initialization of the slipping cone constraint matrix C
    # Simplified friction condition with checks only along X and Y axes
    # For instance C = np.array([[1, 0, -nu], [-1, 0, -nu], [0, 1, -nu], [0, -1, -nu]])
    # To check that abs(Fx) <= (nu * Fz) and abs(Fy) <= (nu * Fz)
    nu = 3

    # C_row, C_col and C_data satisfy the relationship C[C_row[k], C_col[k]] = C_data[k]
    C_row = np.array([0, 1, 2, 3] * 2)
    C_col = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    C_data = np.array([1, -1, 1, -1, -nu, -nu, -nu, -nu])

    # Cumulative number of footholds. For instance if two feet touch the ground during 10
    # steps then 4 feets during 6 steps then nb_tot = 2 * 10 + 4 * 6
    nb_tot = np.sum(n_contacts)

    # Matrix M used for the equality constraints (M.X = N)
    # with dimensions (nb_xf * n_x, nb_xf * n_x + nb_tot * n_f)
    # nb_xf * n_x rows for the constraints x(k+1) = A * x(k) + B * f(k) + g. M is basically
    # [ -1  0  0  0  B  0  0  0
    #    A -1  0  0  0  B  0  0
    #    0  A -1  0  0  0  B  0
    #    0  0  A -1  0  0  0  B ] so nb_of_timesteps * nb_of_states lines

    # X vector is [X1 X2 X3 X4 F0 F1 F2 F3] with X1 = A(0) * X0 + B(0) * F(0)
    # A(0) being the current state of the robot
    # So we have nb_xf * n_x columns to store the Xi and nb_tot * n_f columns to store the Fi

    t_test_diff = clock() - t_test
    print("Initialization stuff:", t_test_diff)
    t_test = clock()
    
    # FILL STATIC PART OF MATRIX M

    # M_row, _col and _data satisfy the relationship M[M_row[k], M_col[k]] = M_data[k]
    M_row = np.array([], dtype=np.int64)
    M_col = np.array([], dtype=np.int64)
    M_data = np.array([], dtype=np.float64)

    sum_contacts_M = 3*np.sum(n_contacts[:, 0])
    M_row2 = np.zeros((nb_xf*n_x + 15*(nb_xf-1) + sum_contacts_M, ), dtype=np.int64)
    M_col2 = np.zeros((nb_xf*n_x + 15*(nb_xf-1) + sum_contacts_M, ), dtype=np.int64)
    M_data2 = np.zeros((nb_xf*n_x + 15*(nb_xf-1) + sum_contacts_M, ), dtype=np.float64)
    M_pt = 0
    print(M_data2.shape)

    # Fill M with minus identity matricesDo !
    M_row2[M_pt:(M_pt+nb_xf*n_x)] = np.arange(0, nb_xf*n_x, 1)
    M_col2[M_pt:(M_pt+nb_xf*n_x)] = np.arange(0, nb_xf*n_x, 1)
    M_data2[M_pt:(M_pt+nb_xf*n_x)] = - np.ones((nb_xf*n_x,))
    M_pt += nb_xf*n_x

    # A_row, A_col and A_data satisfy the relationship A[A_row[k], A_col[k]] = A_data[k]
    A_row = np.array([i for i in range(12)] + [0, 1, 2])
    A_col = np.array([i for i in range(12)] + [6, 7, 8])
    A_data = np.array([1 for i in range(12)] + [dt, dt, dt])

    # Fill M with static part of A(k) matrices
    """M_row = np.hstack((M_row, np.tile(A_row, (nb_xf-1,)) + np.repeat(np.arange(n_x, n_x*(nb_xf), n_x), 15)))
    M_col = np.hstack((M_col, np.tile(A_col, (nb_xf-1,)) + np.repeat(np.arange(0, n_x*(nb_xf-1), n_x), 15)))
    M_data = np.hstack((M_data, np.tile(A_data, (nb_xf-1,))))"""
    M_row2[M_pt:(M_pt+15*(nb_xf-1))] = np.tile(A_row, (nb_xf-1,)) + np.repeat(np.arange(n_x, n_x*(nb_xf), n_x), 15)
    M_col2[M_pt:(M_pt+15*(nb_xf-1))] = np.tile(A_col, (nb_xf-1,)) + np.repeat(np.arange(0, n_x*(nb_xf-1), n_x), 15)
    M_data2[M_pt:(M_pt+15*(nb_xf-1))] = np.tile(A_data, (nb_xf-1,))
    M_pt += 15*(nb_xf-1)

    t_test_diff = clock() - t_test
    print("Fill A in M:", t_test_diff)
    t_test = clock()
    
    # Matrix L used for the equality constraints (L.X <= K)
    # with dimensions (nb_tot * 5, nb_xf * n_x + nb_tot * n_f)
    # nb_tot * 4 rows for the slipping constraints nu fz > abs(fx) and nu fz > abs(fy) (see C matrix)
    # nb_tot rows for the ground reaction constraints fz > 0
    # L is basically a lot of C and -1 stacked depending on the number of footholds during each timestep.
    # The basic bloc is [ 1  0  -nu
    #                    -1  0  -nu
    #                     0  1  -nu
    #                     0 -1  -nu
    #                     0  0   -1 ] for one foothold

    # L_row, _col and _data satisfy the relationship L[L_row[k], L_col[k]] = L_data[k]
    L_row = np.array([], dtype=np.int64)
    L_col = np.array([], dtype=np.int64)
    L_data = np.array([], dtype=np.float64)

    sum_contacts_L = np.sum(n_contacts)
    L_row2 = np.zeros(((C_data.shape[0]+1)*sum_contacts_L , ), dtype=np.int64)
    L_col2 = np.zeros(((C_data.shape[0]+1)*sum_contacts_L , ), dtype=np.int64)
    L_data2 = np.zeros(((C_data.shape[0]+1)*sum_contacts_L, ), dtype=np.float64)
    L_pt = 0
    print("L: ", L_data2.shape)

    # Fill M with B(k) matrices
    # and fill L with slipping cone constraints
    # and fill L with ground reaction force constraints (fz > 0)
    nb_tot = 0
    n_tmp = np.sum(n_contacts)
    S_prev = S[0, :]

    # B_row = np.zeros((3*nb_contacts*nb_xf,))
    # B_col = np.zeros((3*nb_contacts*nb_xf,))
    # B_data = np.zeros((3*nb_contacts*nb_xf,))
    # B_row[(3*nb_contacts*i):(3*nb_contacts*(i+1))] = np.hstack((B_row, np.tile(np.array([6, 7, 8]), (nb_contacts,))))
    # B_col[(3*nb_contacts*i):(3*nb_contacts*(i+1))] = np.hstack((B_col, np.arange(0, 3*nb_contacts, 1)))
    # B_data[(3*nb_contacts*i):(3*nb_contacts*(i+1))] = np.hstack((B_data, dt / m * np.ones((3*nb_contacts,))))

    """L_row_tmp = [0]*(C_data.shape[0]+1)*sum_contacts_L
    L_col_tmp = [0]*(C_data.shape[0]+1)*sum_contacts_L
    L_data_tmp = [0]*(C_data.shape[0]+1)*sum_contacts_L"""
    print(clock() - t_test)
    for i in range(nb_xf):
        # Number of feet touching the ground during this timestep
        nb_contacts = n_contacts[i, 0]

        # B(k) matrix related to x(k+1) = A * x(k) + B * f(k) + g
        # n_x rows (number of row of x) and n_f * nb_contacts columns (depends on the number of footholds)

        # B_row, _col and _data satisfy the relationship B[B_row[k], B_col[k]] = B_data[k]
        # B_row = np.array([], dtype=np.int64)
        # B_col = np.array([], dtype=np.int64)
        # B_data = np.array([], dtype=np.float64)

        # B_row = np.hstack((B_row, np.tile(np.array([6, 7, 8]), (nb_contacts,))))
        # B_col = np.hstack((B_col, np.arange(0, 3*nb_contacts, 1)))
        # B_data = np.hstack((B_data, dt / m * np.ones((3*nb_contacts,))))

        B_row = np.tile(np.array([6, 7, 8], dtype=np.int64), (nb_contacts,))
        B_col = np.arange(0, 3*nb_contacts, 1, dtype=np.int64)
        B_data = dt / m * np.ones((3*nb_contacts,), dtype=np.float64)

        # Filling M with B(k) associated to timestep k. In numpy style:
        # M[(i*n_x):((i+1)*n_x), (nb_xf*n_x+n_f*nb_tot):(nb_xf*n_x+n_f*(nb_tot+nb_contacts))] = B

        """M_row = np.hstack((M_row, B_row + (i*n_x)))
        M_col = np.hstack((M_col, B_col + (nb_xf*n_x+n_f*nb_tot)))
        M_data = np.hstack((M_data, B_data))"""
        M_row2[M_pt:(M_pt+3*nb_contacts)] = B_row + (i*n_x)
        M_col2[M_pt:(M_pt+3*nb_contacts)] = B_col + (nb_xf*n_x+n_f*nb_tot)
        M_data2[M_pt:(M_pt+3*nb_contacts)] = B_data
        M_pt += 3*nb_contacts

        # Filling L with slipping cone constraints and ground reaction force constraints
        """L_row = np.hstack((L_row, np.tile(C_row, (nb_contacts,)) + np.repeat(4 *
                                                                             nb_tot + np.arange(0, 4 * nb_contacts, 4), len(C_row))))
        L_col = np.hstack((L_col, np.tile(C_col, (nb_contacts,)) + np.repeat(nb_xf*n_x +
                                                                             n_f*nb_tot + np.arange(0, n_f * nb_contacts, n_f), len(C_col))))
        L_data = np.hstack((L_data, np.tile(C_data, (nb_contacts,))))"""
        L_row2[L_pt:(L_pt+C_data.shape[0]*nb_contacts)] = np.tile(C_row, (nb_contacts,)) + np.repeat(4 *
                                                                             nb_tot + np.arange(0, 4 * nb_contacts, 4), len(C_row))
        L_col2[L_pt:(L_pt+C_data.shape[0]*nb_contacts)] = np.tile(C_col, (nb_contacts,)) + np.repeat(nb_xf*n_x +
                                                                             n_f*nb_tot + np.arange(0, n_f * nb_contacts, n_f), len(C_col))
        L_data2[L_pt:(L_pt+C_data.shape[0]*nb_contacts)] = np.tile(C_data, (nb_contacts,))
        L_pt += C_data.shape[0]*nb_contacts

        """L_row_tmp[L_pt:(L_pt+C_data.shape[0]*nb_contacts)] = (np.ndarray.tolist(np.tile(C_row, (nb_contacts,)) + np.repeat(4 *
                                                                             nb_tot + np.arange(0, 4 * nb_contacts, 4), len(C_row))))
        L_col_tmp[L_pt:(L_pt+C_data.shape[0]*nb_contacts)] = (np.ndarray.tolist(np.tile(C_col, (nb_contacts,)) + np.repeat(nb_xf*n_x +
                                                                             n_f*nb_tot + np.arange(0, n_f * nb_contacts, n_f), len(C_col))))
        L_data_tmp[L_pt:(L_pt+C_data.shape[0]*nb_contacts)] = (np.ndarray.tolist(np.tile(C_data, (nb_contacts,))))
        L_pt += C_data.shape[0]*nb_contacts"""

        """L_row = np.hstack((L_row, n_tmp * 4 + nb_tot + np.arange(0, nb_contacts, 1)))
        L_col = np.hstack((L_col, nb_xf*n_x+n_f*nb_tot+(n_f-1) + np.arange(0, n_f*nb_contacts, n_f)))
        L_data = np.hstack((L_data, - np.ones((nb_contacts,))))"""
        L_row2[L_pt:(L_pt+nb_contacts)] = n_tmp * 4 + nb_tot + np.arange(0, nb_contacts, 1)
        L_col2[L_pt:(L_pt+nb_contacts)] = nb_xf*n_x+n_f*nb_tot+(n_f-1) + np.arange(0, n_f*nb_contacts, n_f)
        L_data2[L_pt:(L_pt+nb_contacts)] = - np.ones((nb_contacts,))
        L_pt += nb_contacts
        """L_row_tmp[L_pt:(L_pt+nb_contacts)] = (np.ndarray.tolist(n_tmp * 4 + nb_tot + np.arange(0, nb_contacts, 1)))
        L_col_tmp[L_pt:(L_pt+nb_contacts)] = (np.ndarray.tolist(nb_xf*n_x+n_f*nb_tot+(n_f-1) + np.arange(0, n_f*nb_contacts, n_f)))
        L_data_tmp[L_pt:(L_pt+nb_contacts)] = (np.ndarray.tolist(- np.ones((nb_contacts,))))
        L_pt += nb_contacts"""

        # Cumulative number of footholds during the previous step to fill M at the correct place
        nb_tot += nb_contacts

    """L_row2 = np.asarray(L_row_tmp, dtype=np.int64)
    L_col2 = np.asarray(L_col_tmp, dtype=np.int64)
    L_data2 = np.asarray(L_data_tmp, dtype=np.float64)"""
    #print("New:",np.array_equal(L_data3,L_data2))

    t_test_diff = clock() - t_test
    print("Create L and M:", t_test_diff)
    t_test = clock()
    
    # Matrix N on the other side of the equal sign (M.X = N)
    # n_x * nb_xf rows since there is nb_xf equations A * X + B * F + g
    # Only 1 column
    # N = - g [0 0 0 0 0 0 0 0 1 0 0 0]^T - A*X0 + ( - A*X0ref + X1ref) for the first row
    # N = - g [0 0 0 0 0 0 0 0 1 0 0 0]^T        + ( - A*Xkref + Xk+1ref) for the other rows

    # N_row, _col and _data satisfy the relationship N[N_row[k], N_col[k]] = N_data[k]
    # Static part of N is just gravity
    N_row = np.arange(0, nb_xf, 1, dtype=np.int64)*n_x + 8
    N_col = np.zeros((nb_xf,), dtype=np.int64)
    N_data = 9.81*np.ones((nb_xf,), dtype=np.float64)

    # Matrix K on the other side of the inequal sign (L.X <= K)
    # np.sum(n_contacts) * 5 rows to be consistent with L, all coefficients are 0

    # K_row, _col and _data satisfy the relationship K[K_row[k], K_col[k]] = K_data[k]
    K_row = np.array([], dtype=np.int64)
    K_col = np.array([], dtype=np.int64)
    K_data = np.array([], dtype=np.float64)

    t_test_diff = clock() - t_test
    print("Create N and K:", t_test_diff)
    t_test = clock()

    # Convert _row, _col and _data into Compressed Sparse Column matrices (Scipy)
    M_csc = scipy.sparse.csc.csc_matrix((M_data2, (M_row2, M_col2)), shape=(nb_xf * n_x, nb_xf * n_x + nb_tot * n_f))
    N_csc = scipy.sparse.csc.csc_matrix((N_data, (N_row, N_col)), shape=(n_x*nb_xf, 1))
    L_csc = scipy.sparse.csc.csc_matrix((L_data2, (L_row2, L_col2)), shape=(nb_tot * 5, nb_xf * n_x + nb_tot * n_f))
    K_csc = scipy.sparse.csc.csc_matrix((K_data, (K_row, K_col)), shape=(n_tmp*5, 1))

    t_test_diff = clock() - t_test
    print("Conversion to Csc:", t_test_diff)

    print("Shapes: ")
    print(M_data.shape)
    print(N_data.shape)
    print(L_data.shape)
    print(K_data.shape)
    print("M_pt:", M_pt)
    print(np.array_equal(M_data,M_data2))
    print(np.array_equal(L_data,L_data2))
    return M_csc, N_csc, L_csc, K_csc    
    

def createSparseConstraintsMatrices(n_x, n_f, dt, S, n_contacts, footholds, footholds_lock, footholds_no_lock, xref, x0, solo, k_loop, q_w, href):

    t_test = clock()

    footholds_m = footholds.copy()
    """update = (S[0, :] == 0) & (S[-1, :] == 1)  # Detect if one of the feet is in swing phase
    if np.any(update):  # If any foot is in swing phase
        # Get the future position of footholds
        future_footholds = update_target_footholds_no_lock(
            xref[6:12, 0:1], xref[0:6, 0:1], x0[6:12, 0:1], settings.t_stance, S, dt, settings.T_gait, h=href, k=0.03)
        for up in range(update.shape[1]):
            if (update[0, up] == True):  # Considering only feet that need to be updated
                # Updating position of the foothold for this leg.
                footholds_m[:, up:(up+1)] = future_footholds[0:2, up:(up+1)] # no need if only one period in the prediction horizon"""
    update = np.array(S[0]).ravel() == 0
    if np.any(update):
        footholds_m[:, update] = footholds_lock[:, update]

    # Number of timesteps in the prediction horizon
    nb_xf = n_contacts.shape[0]

    # Inertia matrix of the robot in body frame (found in urdf)
    gI = np.diag([0.00578574, 0.01938108, 0.02476124])

    # Inverting the inertia matrix in the global frame
    # R_gI = getRotMatrix(xref[3:6, 0:1])
    # gI_inv = np.linalg.inv(R_gI * gI)

    # Inverting the inertia matrix in local frame
    gI_inv = np.linalg.inv(gI)

    # Mass of the quadruped in [kg] (found in urdf)
    m = 2.2

    # Initialization of the constant - identity matrix
    minusI = - np.eye(n_x)

    # Initialization of the constant part of matrix A
    A = np.zeros((12, 12))
    A[0:3, 0:3] = np.eye(3)
    A[0:3, 6:9] = dt * np.eye(3)
    A[3:6, 3:6] = np.eye(3)
    A[6:9, 6:9] = np.eye(3)
    A[9:12, 9:12] = np.eye(3)

    # A_row, A_col and A_data satisfy the relationship A[A_row[k], A_col[k]] = A_data[k]
    A_row = np.array([i for i in range(12)] + [0, 1, 2] + [3, 4, 5])
    A_col = np.array([i for i in range(12)] + [6, 7, 8] + [9, 10, 11])
    A_data = np.array([1 for i in range(12)] + [dt, dt, dt] + [dt, dt, dt])

    # Initialization of the slipping cone constraint matrix C
    # Simplified friction condition with checks only along X and Y axes
    # For instance C = np.array([[1, 0, -nu], [-1, 0, -nu], [0, 1, -nu], [0, -1, -nu]])
    # To check that abs(Fx) <= (nu * Fz) and abs(Fy) <= (nu * Fz)
    nu = 2

    # C_row, C_col and C_data satisfy the relationship C[C_row[k], C_col[k]] = C_data[k]
    C_row = np.array([0, 1, 2, 3] * 2)
    C_col = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    C_data = np.array([1, -1, 1, -1, -nu, -nu, -nu, -nu])

    # Cumulative number of footholds. For instance if two feet touch the ground during 10
    # steps then 4 feets during 6 steps then nb_tot = 2 * 10 + 4 * 6
    nb_tot = np.sum(n_contacts)

    # Matrix M used for the equality constraints (M.X = N)
    # with dimensions (nb_xf * n_x, nb_xf * n_x + nb_tot * n_f)
    # nb_xf * n_x rows for the constraints x(k+1) = A * x(k) + B * f(k) + g. M is basically
    # [ -1  0  0  0  B  0  0  0
    #    A -1  0  0  0  B  0  0
    #    0  A -1  0  0  0  B  0
    #    0  0  A -1  0  0  0  B ] so nb_of_timesteps * nb_of_states lines

    # X vector is [X1 X2 X3 X4 F0 F1 F2 F3] with X1 = A(0) * X0 + B(0) * F(0)
    # A(0) being the current state of the robot
    # So we have nb_xf * n_x columns to store the Xi and nb_tot * n_f columns to store the Fi

    t_test_diff = clock() - t_test
    print("Initialization stuff:", t_test_diff)

    t_test = clock()

    # M_row, _col and _data satisfy the relationship M[M_row[k], M_col[k]] = M_data[k]
    M_row = np.array([], dtype=np.int64)
    M_col = np.array([], dtype=np.int64)
    M_data = np.array([], dtype=np.float64)

    # Fill M with minus identity matrices
    M_row = np.arange(0, nb_xf*n_x, 1)
    M_col = np.arange(0, nb_xf*n_x, 1)
    M_data = - np.ones((nb_xf*n_x,))

    # Fill M with A(k) matrices
    # Looped version:
    """ for i in range(nb_xf-1):
        # Dynamic part of A is related to the dt * R term
        c, s = np.cos(xref[5, i]), np.sin(xref[5, i])
        # R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        # A[3:6, 9:12] = dt * R
        # Then we just put A in M, here in numpy style
        # M[((i+1)*n_x):((i+2)*n_x), (i*n_x):((i+1)*n_x)] = A

        M_row = np.hstack((M_row, A_row + ((i+1)*n_x)))
        M_col = np.hstack((M_col, A_col + (i*n_x)))
        M_data = np.hstack((M_data, A_data))

        M_row = np.hstack((M_row, np.array([3,  3,  4,  4,  5]) + ((i+1)*n_x)))
        M_col = np.hstack((M_col, np.array([9, 10,  9, 10, 11]) + (i*n_x)))
        M_data = np.hstack((M_data, dt * np.array([c, s, -s, c, 1])))"""

    # Non-looped version:
    M_row = np.hstack((M_row, np.tile(A_row, (nb_xf-1,)) + np.repeat(np.arange(n_x, n_x*(nb_xf), n_x), 18)))
    M_col = np.hstack((M_col, np.tile(A_col, (nb_xf-1,)) + np.repeat(np.arange(0, n_x*(nb_xf-1), n_x), 18)))
    M_data = np.hstack((M_data, np.tile(A_data, (nb_xf-1,))))

    # Rotation is not required anymore since we are working in local frame
    """c_t = np.cos(xref[5:6, 0:(nb_xf-1)])
    s_t = np.sin(xref[5:6, 0:(nb_xf-1)])

    M_row = np.hstack((M_row, np.tile(np.array([3,  3,  4,  4,  5]),
                                      (nb_xf-1,)) + np.repeat(np.arange(n_x, n_x*(nb_xf), n_x), 5)))
    M_col = np.hstack((M_col, np.tile(np.array([9, 10,  9, 10, 11]),
                                      (nb_xf-1,)) + np.repeat(np.arange(0, n_x*(nb_xf-1), n_x), 5)))
    M_data = np.hstack(
        (M_data, dt * np.reshape(np.vstack((c_t, s_t, -s_t, c_t, np.ones((1, c_t.shape[1])))), (-1,), order='F')))"""

    t_test_diff = clock() - t_test
    print("Fill A in M:", t_test_diff)

    # Matrix L used for the equality constraints (L.X <= K)
    # with dimensions (nb_tot * 5, nb_xf * n_x + nb_tot * n_f)
    # nb_tot * 4 rows for the slipping constraints nu fz > abs(fx) and nu fz > abs(fy) (see C matrix)
    # nb_tot rows for the ground reaction constraints fz > 0
    # L is basically a lot of C and -1 stacked depending on the number of footholds during each timestep.
    # The basic bloc is [ 1  0  -nu
    #                    -1  0  -nu
    #                     0  1  -nu
    #                     0 -1  -nu
    #                     0  0   -1 ] for one foothold

    # L_row, _col and _data satisfy the relationship L[L_row[k], L_col[k]] = L_data[k]
    L_row = np.array([], dtype=np.int64)
    L_col = np.array([], dtype=np.int64)
    L_data = np.array([], dtype=np.float64)

    if k_loop == 22:
        passa = 1

    # Fill M with B(k) matrices
    # and fill L with slipping cone constraints
    # and fill L with ground reaction force constraints (fz > 0)
    nb_tot = 0
    n_tmp = np.sum(n_contacts)
    S_prev = S[0, :].copy()
    for i in range(nb_xf):
        update = (S_prev == 1) & (S[i, :] == 0)  # Detect if one of the feet just left the ground
        if np.any(update):
            for up in range(update.shape[1]):
                if (update[0, up] == True):  # Considering only feet that just left the ground
                    # Updating position of the foothold for this leg.
                    footholds_m[:, up:(up+1)] = footholds_no_lock[0:2, up:(up+1)]  # no need if only one period in the prediction horizon

        if False and np.any(update):  # If any foot left the ground (start of swing phase)
            # Get the future position of footholds
            print("UPDATE")
            S_tmp = np.vstack((settings.S[i:, :], settings.S[0:i, :]))
            print(xref[6:12, i:(i+1)])
            print(xref[0:6, i:(i+1)])
            print(x0[6:12, 0:1])
            future_footholds = update_target_footholds_no_lock(
                xref[6:12, i:(i+1)], xref[0:6, i:(i+1)], x0[6:12, 0:1], settings.t_stance, S_tmp, dt, settings.T_gait, h=x0[2, 0], k=0.03)
            print("Futur footholds local")
            print(future_footholds)

            # As the output of update_target_footholds_no_lock is in local frame then
            # the future position and orientation has to be taken into account to be in optimisation frame
            indexes = (S_tmp!=0).argmax(axis=0)

            for j in range(S_tmp.shape[1]):
                if (i+indexes[0,j]) < S.shape[0]:
                    c, s = np.cos(xref[5, i+indexes[0,j]]), np.sin(xref[5, i+indexes[0,j]])
                    R = np.array([[c, -s], [s, c]])
                    future_footholds[:,j] = xref[0:2, (i+indexes[0,j])] + np.dot(R, future_footholds[:,j])
            # c, s = np.cos(xref[5, i]), np.sin(xref[5, i])
            # R = np.array([[c, -s], [s, c]])
            #future_footholds = np.tile(xref[0:2, i:(i+1)], (1, 4)) + np.dot(R, future_footholds)

            for up in range(update.shape[1]):
                if (update[0, up] == True):  # Considering only feet that just touched the ground
                    # Updating position of the foothold for this leg.
                    footholds_m[:, up:(up+1)] = future_footholds[0:2, up:(up+1)] # no need if only one period in the prediction horizon
        # Saving current state of feet (touching or not) for the next step
        S_prev = (S[i, :]).copy()

        # Number of feet touching the ground during this timestep
        nb_contacts = n_contacts[i, 0]

        # B(k) matrix related to x(k+1) = A * x(k) + B * f(k) + g
        # n_x rows (number of row of x) and n_f * nb_contacts columns (depends on the number of footholds)

        # B_row, _col and _data satisfy the relationship B[B_row[k], B_col[k]] = B_data[k]
        B_row = np.array([], dtype=np.int64)
        B_col = np.array([], dtype=np.int64)
        B_data = np.array([], dtype=np.float64)

        # Position of footholds in the global frame
        pos_contacts = footholds_m[:, (S[i, :] == 1).getA()[0, :]]
        #print(pos_contacts)
        # For each foothold during this timestep
        """for j in range(nb_contacts):
            # Relative position of the foothold compared to the center of mass (here center of the base)
            # contact_foot = np.array([[pos_contacts[0, j]], [pos_contacts[1, j]], [0]]) - xref[0:3, i:(i+1)]

            # Filling the B matrix. In numpy style, just like in the paper:
            # B[9:12, (3*j):(3*j+3)] = dt * np.dot(gI_inv, getSkew(contact_foot))
            # B[6:9, (3*j):(3*j+3)] = dt / m * np.eye(3)

            # Looped version of filling B
            tmp = dt * np.dot(gI_inv, getSkew(contact_foot))
            B_row = np.hstack((B_row, np.array([9, 9, 9, 10, 10, 10, 11, 11, 11])))
            B_col = np.hstack((B_col, np.tile(np.arange(3*j, 3*j+3, 1), (3,))))
            B_data = np.hstack((B_data, tmp.reshape((-1,))))

            B_row = np.hstack((B_row, np.array([6, 7, 8])))
            B_col = np.hstack((B_col, np.arange(3*j, 3*j+3, 1)))
            B_data = np.hstack((B_data, dt / m * np.ones((3,))))

            # Filling the L matrix as explained above. In numpy style:
            # L[(4*nb_tot+4*j):(4*nb_tot+4*(j+1)), (nb_xf*n_x+n_f*(nb_tot+j)):(nb_xf*n_x+n_f*(nb_tot+(j+1)))] = C
            # L[(n_tmp * 4 + nb_tot + j), (nb_xf*n_x+n_f*(nb_tot+j)+(n_f-1))] = -1

            # Looped version of filling L
            L_row = np.hstack((L_row, C_row + (4*nb_tot+4*j)))
            L_col = np.hstack((L_col, C_col + (nb_xf*n_x+n_f*(nb_tot+j))))
            L_data = np.hstack((L_data, C_data))

            L_row = np.hstack((L_row, (n_tmp * 4 + nb_tot + j)))
            L_col = np.hstack((L_col, (nb_xf*n_x+n_f*(nb_tot+j)+(n_f-1))))
            L_data = np.hstack((L_data, -1))"""

        # Non-looped version of filling B
        contact_foot = np.vstack((pos_contacts, np.zeros((1, pos_contacts.shape[1])))) - xref[0:3, i:(i+1)]
        #print("Footholds_m: \n", footholds_m)
        
        """print("### Pos_contacts:")
        print(pos_contacts)
        print(contact_foot)
        print(xref[0:3, i:(i+1)].transpose())"""

        enable_support_lines = False
        if enable_support_lines:
            if (k_loop == 22) and (i == 24):
                debug = 1
            if (i==0):
                for j_c in range(nb_xf):
                    if j_c % 1 == 0:
                        for i_c in range(4):
                            solo.viewer.gui.addCurve("world/support_"+str(j_c)+"_"+str(i_c), [[0., 0., 0.],[0.,0.,0.]], [0.0, 0.0, 0.0, 0.0])
                            #solo.viewer.gui.setCurveLineWidth("world/support_"+str(i)+"_"+str(i_c), 0.0)
                            #solo.viewer.gui.setColor("world/support_"+str(i)+"_"+str(i_c), [0.0,0.0,0.0,0.0])
                solo.viewer.gui.refresh()

            if (i % 1) == 0:
                
                num_foot = 0
                for i_c in range(4):
                    if S[i, i_c] == 1:
                        c, s = np.cos(q_w[5, 0]), np.sin(q_w[5, 0])
                        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 0]])
                        curvePoints_1 = np.dot(R, xref[0:3, i:(i+1)]) + q_w[0:3, 0:1] + np.array([[0],[0],[0.05]])
                        curvePoints_2 = np.dot(R[0:2, 0:2], pos_contacts) + np.tile(q_w[0:2, 0:1],(1,pos_contacts.shape[1]))
                        solo.viewer.gui.addCurve("world/support_"+str(i)+"_"+str(i_c), [[0., 0., 0.],[1.,1.,1]], [i/nb_xf, 0.0, i/nb_xf, 0.5])
                        # solo.viewer.gui.setCurvePoints("world/support_"+str(i)+"_"+str(i_c), [xref[0:3, i].tolist(), [pos_contacts[0,num_foot],pos_contacts[1,num_foot],0.]])
                        solo.viewer.gui.setCurvePoints("world/support_"+str(i)+"_"+str(i_c), [curvePoints_1[:,0].tolist(), [curvePoints_2[0, num_foot], curvePoints_2[1, num_foot], 0.]])
                        solo.viewer.gui.setCurveLineWidth("world/support_"+str(i)+"_"+str(i_c), 8.0)
                        solo.viewer.gui.setColor("world/support_"+str(i)+"_"+str(i_c), [i/nb_xf, 0.0, i/nb_xf, 0.5])
                        num_foot += 1
                    else:
                        solo.viewer.gui.setCurveLineWidth("world/support_"+str(i)+"_"+str(i_c), 0.0)
                        solo.viewer.gui.setColor("world/support_"+str(i)+"_"+str(i_c), [0.0,0.0,0.0,0.0])
                solo.viewer.gui.refresh()

        # tmp = np.reshape(np.array([np.zeros((nb_contacts,)), -contact_foot[2, :], contact_foot[1, :],
        #                  contact_foot[2, :], np.zeros((nb_contacts,)), -contact_foot[0, :],
        #                  -contact_foot[1, :], contact_foot[0, :], np.zeros(nb_contacts,)]), (-1,), order='F')
        tmp = dt * np.dot(gI_inv, np.array([[np.zeros((nb_contacts,)), -contact_foot[2, :], contact_foot[1, :]],
                                            [contact_foot[2, :], np.zeros((nb_contacts,)), -contact_foot[0, :]],
                                            [-contact_foot[1, :], contact_foot[0, :], np.zeros(nb_contacts,)]]))
        #print("Contact_foot: \n", contact_foot)
        B_row = np.hstack((B_row, np.tile(np.array([9, 9, 9, 10, 10, 10, 11, 11, 11]), (nb_contacts,))))
        B_col = np.hstack((B_col, np.tile(np.array([0, 1, 2]), (3*nb_contacts,)
                                          ) + np.repeat(3 * np.arange(0, nb_contacts, 1), 9)))
        B_data = np.hstack((B_data, np.moveaxis(-tmp, 2, 0).reshape((-1,))))

        B_row = np.hstack((B_row, np.tile(np.array([6, 7, 8]), (nb_contacts,))))
        B_col = np.hstack((B_col, np.arange(0, 3*nb_contacts, 1)))
        B_data = np.hstack((B_data, dt / m * np.ones((3*nb_contacts,))))

        # Non-looped version of filling L
        L_row = np.hstack((L_row, np.tile(C_row, (nb_contacts,)) + np.repeat(4 *
                                                                             nb_tot + np.arange(0, 4 * nb_contacts, 4), len(C_row))))
        L_col = np.hstack((L_col, np.tile(C_col, (nb_contacts,)) + np.repeat(nb_xf*n_x +
                                                                             n_f*nb_tot + np.arange(0, n_f * nb_contacts, n_f), len(C_col))))
        L_data = np.hstack((L_data, np.tile(C_data, (nb_contacts,))))

        L_row = np.hstack((L_row, n_tmp * 4 + nb_tot + np.arange(0, nb_contacts, 1)))
        L_col = np.hstack((L_col, nb_xf*n_x+n_f*nb_tot+(n_f-1) + np.arange(0, n_f*nb_contacts, n_f)))
        L_data = np.hstack((L_data, - np.ones((nb_contacts,))))

        # Filling M with B(k) associated to timestep k. In numpy style:
        # M[(i*n_x):((i+1)*n_x), (nb_xf*n_x+n_f*nb_tot):(nb_xf*n_x+n_f*(nb_tot+nb_contacts))] = B

        M_row = np.hstack((M_row, B_row + (i*n_x)))
        M_col = np.hstack((M_col, B_col + (nb_xf*n_x+n_f*nb_tot)))
        M_data = np.hstack((M_data, B_data))

        # Cumulative number of footholds during the previous step to fill M and L at the correct place
        nb_tot += nb_contacts

    # Matrix N on the other side of the equal sign (M.X = N)
    # n_x * nb_xf rows since there is nb_xf equations A * X + B * F + g
    # Only 1 column
    # N = - g [0 0 0 0 0 0 0 0 1 0 0 0]^T - A*X0 + ( - A*X0ref + X1ref) for the first row
    # N = - g [0 0 0 0 0 0 0 0 1 0 0 0]^T        + ( - A*Xkref + Xk+1ref) for the other rows

    t_test_diff = clock() - t_test
    print("Create L and M:", t_test_diff)

    t_test = clock()

    N = np.zeros((n_x*nb_xf, 1))

    # N_row, _col and _data satisfy the relationship N[N_row[k], N_col[k]] = N_data[k]
    N_row = np.array([], dtype=np.int64)
    N_col = np.array([], dtype=np.int64)
    N_data = np.array([], dtype=np.float64)

    # Matrix K on the other side of the inequal sign (L.X <= K)
    # np.sum(n_contacts) * 5 rows to be consistent with L, all coefficients are 0

    # K_row, _col and _data satisfy the relationship K[K_row[k], K_col[k]] = K_data[k]
    K_row = np.array([], dtype=np.int64)
    K_col = np.array([], dtype=np.int64)
    K_data = np.array([], dtype=np.float64)

    # The gravity vector is included in N
    # It has an effect on the 8th coefficient of the state vector for each timestep (linear velocity along Z)
    # In numpy style:
    g = np.zeros((n_x, 1))
    g[8, 0] = -9.81*dt
    for i in range(nb_xf):
        N[n_x*i:(n_x*(i+1)), 0:1] = - g

    # Including - A*X0 in the first row of N
    c, s = np.cos(xref[5, 0]), np.sin(xref[5, 0])
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    A[3:6, 9:12] = dt * R
    # Numpy style: N[0:n_x, 0:1] += np.dot(A, - x0)
    N[0:n_x, 0:1] += np.dot(A, - x0)

    # Here we include both the gravity and -A*X0 at the same time
    tmp = np.dot(A, - x0)
    tmp = np.vstack((tmp, np.zeros((n_x*(nb_xf-1), 1))))
    tmp[np.arange(0, nb_xf, 1)*n_x + 8, 0:1] += 9.81*dt

    N_row = np.hstack((N_row, np.arange(0, n_x*nb_xf, 1)))
    N_col = np.hstack((N_col, np.zeros((n_x*nb_xf,), dtype=np.int64)))
    N_data = np.hstack((N_data, tmp.reshape((-1,))))

    # D is the third term of the sum of N that includes (- A*Xk-1ref + Xkref). For instance
    # [  1   0   0
    #    A   1   0
    #    0   A   1 ] that will be used to do a matrix product with Xref vector
    D = np.zeros((n_x*nb_xf, n_x*nb_xf))

    # D_row, _col and _data satisfy the relationship D[D_row[k], D_col[k]] = D_data[k]
    D_row = np.array([], dtype=np.int64)
    D_col = np.array([], dtype=np.int64)
    D_data = np.array([], dtype=np.float64)

    # Fill D with identity matrices
    for i in range(nb_xf):
        D[(i*n_x):((i+1)*n_x), (i*n_x):((i+1)*n_x)] = np.eye(n_x)

    D_row = np.arange(0, nb_xf*n_x, 1)
    D_col = np.arange(0, nb_xf*n_x, 1)
    D_data = + np.ones((nb_xf*n_x,))

    # Fill D with A(k) matrices
    for i in range(nb_xf-1):
        # Dynamic part of A
        c, s = np.cos(xref[5, (i+1)]), np.sin(xref[5, (i+1)])
        R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        A[3:6, 9:12] = dt * R
        D[((i+1)*n_x):((i+2)*n_x), (i*n_x):((i+1)*n_x)] = -A

        D_row = np.hstack((D_row, A_row + ((i+1)*n_x)))
        D_col = np.hstack((D_col, A_col + (i*n_x)))
        D_data = np.hstack((D_data, A_data))

        D_row = np.hstack((D_row, np.array([3,  3,  4,  4,  5]) + ((i+1)*n_x)))
        D_col = np.hstack((D_col, np.array([9, 10,  9, 10, 11]) + (i*n_x)))
        D_data = np.hstack((D_data, np.array([c, s, -s, c, 1])))

    
    t_test_diff = clock() - t_test
    print("Create K and N:", t_test_diff)

    t_test = clock()

    # Convert _row, _col and _data into Compressed Sparse Column matrices (Scipy)
    M_csc = scipy.sparse.csc.csc_matrix((M_data, (M_row, M_col)), shape=(nb_xf * n_x, nb_xf * n_x + nb_tot * n_f))
    N_csc = scipy.sparse.csc.csc_matrix((N_data, (N_row, N_col)), shape=(n_x*nb_xf, 1))
    L_csc = scipy.sparse.csc.csc_matrix((L_data, (L_row, L_col)), shape=(nb_tot * 5, nb_xf * n_x + nb_tot * n_f))
    K_csc = scipy.sparse.csc.csc_matrix((K_data, (K_row, K_col)), shape=(n_tmp*5, 1))

    t_test_diff = clock() - t_test
    print("Conversion to Csc:", t_test_diff)

    # Include D*xref to N which already contains - g - A*X0 (first row) or just - g (other rows)
    N = N + np.dot(D, (xref[:, 1:]).reshape((-1, 1), order='F'))
    N_csc += scipy.sparse.csc.csc_matrix(np.dot(D, (xref[:, 1:]).reshape((-1, 1), order='F')), shape=(n_x*nb_xf, 1))

    #print("N: ", np.array_equal(N_csc.toarray(), N))
    # Check if matrices are properly created
    """print("M: ", np.array_equal(M_csc.toarray(), M))
    print("N: ", np.array_equal(N_csc.toarray(), N))
    print("L: ", np.array_equal(L_csc.toarray(), L))
    print("K: ", np.array_equal(K_csc.toarray(), K))"""

    #print(gI_inv)

    return M_csc, N_csc, L_csc, K_csc


def createConstraintsMatrices(n_x, n_f, dt, S, n_contacts, footholds, xref, x0):
    nb_xf = n_contacts.shape[0]

    minusI = - np.eye(n_x)

    gI = np.diag([0.00578574, 0.01938108, 0.02476124])
    R_gI = getRotMatrix(xref[3:6, 0:1])
    gI_inv = np.linalg.inv(R_gI * gI)
    m = 2.2  # mass of the quadruped

    # Initialization of the constant part of matrix A
    A = np.zeros((12, 12))
    A[0:3, 0:3] = np.eye(3)
    A[0:3, 6:9] = dt * np.eye(3)
    A[3:6, 3:6] = np.eye(3)
    A[6:9, 6:9] = np.eye(3)
    A[9:12, 9:12] = np.eye(3)

    # Initialization of the slipping cone constraint matrix C
    nu = 0.7
    C = np.array([[1, 0, -nu], [-1, 0, -nu], [0, 1, -nu], [0, -1, -nu]])

    # nb_xf * n_x rows for the constraints x(k+1) = A * x(k) + B * f(k) + g
    # nb_tot * 2 rows for the slipping constraints nu fz - fxy > 0
    # nb_tot rows for the ground reaction constraints fz > 0
    nb_tot = np.sum(n_contacts)
    # M = np.zeros((nb_xf * n_x, nb_xf * n_x + nb_tot * n_f))
    M = scipy.sparse.lil_matrix((int(nb_xf * n_x), int(nb_xf * n_x + nb_tot * n_f)))

    # Fill M with minus identity matrices
    for i in range(nb_xf):
        M[(i*n_x):((i+1)*n_x), (i*n_x):((i+1)*n_x)] = minusI

    # Fill M with A(k) matrices
    for i in range(nb_xf-1):
        # Dynamic part of A
        c, s = np.cos(xref[5, i]), np.sin(xref[5, i])
        R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        A[3:6, 9:12] = dt * R
        M[((i+1)*n_x):((i+2)*n_x), (i*n_x):((i+1)*n_x)] = A

    # L = np.zeros((nb_tot * 5, nb_xf * n_x + nb_tot * n_f))
    L = scipy.sparse.lil_matrix((int(nb_tot * 5), int(nb_xf * n_x + nb_tot * n_f)))

    # Fill M with B(k) matrices
    # and fill L with slipping cone constraints
    # and fill L with ground reaction force constraints (fz > 0)
    nb_tot = 0
    n_tmp = np.sum(n_contacts)
    S_prev = S[0, :]
    for i in range(nb_xf):
        update = (S_prev == 0) & (S[i, :] == 1)
        if np.any(update):
            future_footholds = update_target_footholds_no_lock(
                xref[6:12, i:(i+1)], xref[0:6, i:(i+1)], xref[6:12, i:(i+1)], settings.t_stance, S, dt, settings.T_gait)
            for up in range(update.shape[0]):
                if up == True:
                    footholds[:, up:(up+1)] = future_footholds[0:2, up:(up+1)]
        S_prev = (S[i, :]).copy()

        nb_contacts = n_contacts[i, 0]
        B = np.zeros((n_x, n_f * nb_contacts))
        pos_contacts = footholds[:, (S[i, :] == 1).getA()[0, :]]

        for j in range(nb_contacts):
            contact_foot = np.array([[pos_contacts[0, j]], [pos_contacts[1, j]], [0]]) - xref[0:3, i:(i+1)]
            B[9:12, (3*j):(3*j+3)] = dt * np.dot(gI_inv, getSkew(contact_foot))
            B[6:9, (3*j):(3*j+3)] = dt / m * np.eye(3)

            L[(4*nb_tot+4*j):(4*nb_tot+4*(j+1)), (nb_xf*n_x+n_f*(nb_tot+j)):(nb_xf*n_x+n_f*(nb_tot+(j+1)))] = C
            L[(n_tmp * 4 + nb_tot + j), (nb_xf*n_x+n_f*(nb_tot+j)+(n_f-1))] = -1

            # M[(nb_xf*n_x+2*i):(nb_xf*n_x+2*(i+1)), (nb_xf*n_x+n_f*i):(nb_xf*n_x+n_f*(i+1))] = C
            # M[(nb_xf * (n_x + 2) + i), (nb_xf*n_x+n_f*i+(n_f-1))] = 1

        M[(i*n_x):((i+1)*n_x), (nb_xf*n_x+n_f*nb_tot):(nb_xf*n_x+n_f*(nb_tot+nb_contacts))] = B

        nb_tot += nb_contacts

    # print(M.shape)
    # print(L.shape)

    # Matrix N on the other side of the equal sign (M.X = N)
    N = np.zeros((n_x*nb_xf, 1))

    # Matrix K on the other side of the inequal sign (L.X <= K)
    K = np.zeros((n_tmp*5, 1))

    g = np.zeros((n_x, 1))
    g[8, 0] = -9.81*dt

    for i in range(nb_xf):
        N[n_x*i:(n_x*(i+1)), 0:1] = - g

    c, s = np.cos(xref[5, 0]), np.sin(xref[5, 0])
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    A[3:6, 9:12] = dt * R
    # N[0:n_x, 0:1] = np.dot(A, -xref[0:n_x, 0:1] - x0)
    N[0:n_x, 0:1] += np.dot(A, - x0)

    D = np.zeros((n_x*nb_xf, n_x*nb_xf))

    # Fill D with minus identity matrices
    for i in range(nb_xf):
        D[(i*n_x):((i+1)*n_x), (i*n_x):((i+1)*n_x)] = -minusI

    # Fill D with A(k) matrices
    for i in range(nb_xf-1):
        # Dynamic part of A
        c, s = np.cos(xref[5, (i+1)]), np.sin(xref[5, (i+1)])
        R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        A[3:6, 9:12] = dt * R
        D[((i+1)*n_x):((i+2)*n_x), (i*n_x):((i+1)*n_x)] = -A

    N = N + np.dot(D, (xref[:, 1:]).reshape((-1, 1), order='F'))

    # print(N.shape)
    # print(K.shape)

    return M, N, L, K


def createConstraintsMatricesBis(n_x, n_f, dt, S, n_contacts, footholds, xref):
    nb_xf = n_contacts.shape[0]

    minusI = - np.eye(n_x)

    gI = np.diag([0.00578574, 0.01938108, 0.02476124])
    R_gI = getRotMatrix(xref[3:6, 0:1])
    gI_inv = np.linalg.inv(R_gI * gI)
    m = 2.2  # mass of the quadruped

    # Initialization of the constant part of matrix A
    A = np.zeros((12, 12))
    A[0:3, 0:3] = np.eye(3)
    A[0:3, 6:9] = dt * np.eye(3)
    A[3:6, 3:6] = np.eye(3)
    A[6:9, 6:9] = np.eye(3)
    A[9:12, 9:12] = np.eye(3)

    # Initialization of the slipping cone constraint matrix C
    nu = 0.7
    C = np.array([[-1, 0, nu], [1, 0, nu], [0, -1, nu], [0, 1, nu]])

    # nb_xf * n_x rows for the constraints x(k+1) = A * x(k) + B * f(k) + g
    # nb_tot * 2 rows for the slipping constraints nu fz - fxy > 0
    # nb_tot rows for the ground reaction constraints fz > 0
    nb_tot = np.sum(n_contacts)
    M = np.zeros((nb_xf * n_x + nb_tot * 5, nb_xf * n_x + nb_tot * n_f))

    # Fill M with minus identity matrices
    for i in range(nb_xf):
        M[(i*n_x):((i+1)*n_x), (i*n_x):((i+1)*n_x)] = minusI

    # Fill M with A(k) matrices
    for i in range(nb_xf-1):
        # Dynamic part of A
        c, s = np.cos(xref[5, i]), np.sin(xref[5, i])
        R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        A[3:6, 9:12] = dt * R
        M[((i+1)*n_x):((i+2)*n_x), (i*n_x):((i+1)*n_x)] = A

    # Fill M with B(k) matrices
    # and fill M with slipping cone constraints
    # and fill M with ground reaction force constraints (fz > 0)
    nb_tot = 0
    n_tmp = np.sum(n_contacts)
    for i in range(nb_xf):
        nb_contacts = n_contacts[i, 0]
        B = np.zeros((n_x, n_f * nb_contacts))
        pos_contacts = footholds[:, (S[i, :] == 1)]  # .getA()[0, :]]

        for j in range(nb_contacts):
            contact_foot = np.array([[pos_contacts[0, j]], [pos_contacts[1, j]], [0]]) - xref[0:3, i:(i+1)]
            B[9:12, (3*j):(3*j+3)] = dt * np.dot(gI_inv, getSkew(contact_foot))
            B[6:9, (3*j):(3*j+3)] = dt / m * np.eye(3)

            M[(nb_xf*n_x+4*nb_tot+4*j):(nb_xf*n_x+4*nb_tot+4*(j+1)),
              (nb_xf*n_x+n_f*(nb_tot+j)):(nb_xf*n_x+n_f*(nb_tot+(j+1)))] = C
            M[(nb_xf * n_x + n_tmp * 4 + nb_tot + j), (nb_xf*n_x+n_f*(nb_tot+j)+(n_f-1))] = 1

            # M[(nb_xf*n_x+2*i):(nb_xf*n_x+2*(i+1)), (nb_xf*n_x+n_f*i):(nb_xf*n_x+n_f*(i+1))] = C
            # M[(nb_xf * (n_x + 2) + i), (nb_xf*n_x+n_f*i+(n_f-1))] = 1

        M[(i*n_x):((i+1)*n_x), (nb_xf*n_x+n_f*nb_tot):(nb_xf*n_x+n_f*(nb_tot+nb_contacts))] = B

        nb_tg = np.zeros((n_x, 1))

    print(M)

    # Matrix N on the other side of the equal sign (M.X = N)

    N = np.zeros((n_x*nb_xf + n_tmp*5, 1))

    g = np.zeros((n_x, 1))
    g[8, 0] = -9.81*dt

    for i in range(nb_xf):
        N[n_x*i:(n_x*(i+1)), 0:1] = g

    D = np.zeros((n_x*nb_xf + n_tmp*5, n_x*nb_xf))

    # Fill D with minus identity matrices
    for i in range(nb_xf):
        D[(i*n_x):((i+1)*n_x), (i*n_x):((i+1)*n_x)] = minusI

    # Fill D with A(k) matrices
    for i in range(nb_xf-1):
        # Dynamic part of A
        c, s = np.cos(xref[5, i]), np.sin(xref[5, i])
        R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        A[3:6, 9:12] = dt * R
        D[((i+1)*n_x):((i+2)*n_x), (i*n_x):((i+1)*n_x)] = A

    N = N + np.dot(D, (xref[:, 1:]).reshape((-1, 1)))

    print(M.shape)
    print(N.shape)

    N_low = N
    N_up = N.copy()

    nb_tot = 0
    for i in range(nb_xf):
        nb_contacts = n_contacts[i, 0]

        for j in range(nb_contacts):
            N_up[nb_xf*n_x+4*nb_tot+4*j, 0] = 100000
            N_low[nb_xf*n_x+4*nb_tot+4*j+1, 0] = -100000
            N_up[nb_xf*n_x+4*nb_tot+4*j+2, 0] = 100000
            N_low[nb_xf*n_x+4*nb_tot+4*j+3, 0] = -100000

        nb_tot += nb_contacts

    return M, N


def checkQPSolution(n_x, n_f, dt, S, n_contacts, footholds, xref, x0, x_qp):

    # Inertia matrix of the robot in body frame (found in urdf)
    gI = np.diag([0.00578574, 0.01938108, 0.02476124])

    # Inverting the inertia matrix in the global frame
    #  R_gI = getRotMatrix(xref[3:6, 0:1])
    # gI_inv = np.linalg.inv(R_gI * gI)
    
    gI_inv = np.linalg.inv(gI)
    #print(gI_inv)

    m = 2.2  # mass of the quadruped

    g = np.zeros((n_x, 1))
    g[8, 0] = -9.81*dt

    nb_xf = n_contacts.shape[0]

    # Check friction cone and ground reaction force
    nb_tot = 0
    n_tmp = np.sum(n_contacts)
    for i in range(nb_xf):
        nb_contacts = n_contacts[i, 0]
        for j in range(nb_contacts):
            fx = x_qp[nb_xf*n_x+3*nb_tot+3*j+0]
            fy = x_qp[nb_xf*n_x+3*nb_tot+3*j+1]
            fz = x_qp[nb_xf*n_x+3*nb_tot+3*j+2]

            nu = 10

            if (fz < -0.0001):
                print("ERROR: Ground normal force cannot be negative.")

            if ((nu*fz+0.0001) < np.abs(fx)) or ((nu*fz+0.0001) < np.abs(fy)):
                print("ERROR: Contact force out of friction cone.")

    A = np.zeros((12, 12))
    A[0:3, 0:3] = np.eye(3)
    A[0:3, 6:9] = dt * np.eye(3)
    A[3:6, 3:6] = np.eye(3)
    A[6:9, 6:9] = np.eye(3)
    A[9:12, 9:12] = np.eye(3)

    # Check if x(k+1) = A x(k) + B f(k) + g is verified
    nb_tot = 0
    for i in range(nb_xf):

        nb_contacts = n_contacts[i, 0]

        #c, s = np.cos(x_qp[(n_x*i+5)]), np.sin(x_qp[(n_x*i+5)])
        c, s = np.cos(xref[5, i:(i+1)]), np.sin(xref[5, i:(i+1)])
        R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        A[3:6, 9:12] = dt * R
        
        B = np.zeros((n_x, n_f * nb_contacts))
        pos_contacts = footholds[:, (S[i, :] == 1).getA()[0, :]]

        for j in range(nb_contacts):
            contact_foot = np.array([[pos_contacts[0, j]], [pos_contacts[1, j]], [0]]) - xref[0:3, i:(i+1)]
            B[9:12, (3*j):(3*j+3)] = dt * np.dot(gI_inv, getSkew(contact_foot))
            B[6:9, (3*j):(3*j+3)] = dt / m * np.eye(3)

        if (i == 0):
            x_next = np.dot(A, x0)
            x_next += np.dot(B, np.matrix(x_qp[(nb_xf*n_x+nb_tot*3):(nb_xf*n_x+nb_tot*3+nb_contacts*3)]).T)
            x_next += g
            x_next -= xref[:, 1:2]
            x_next -= np.matrix(x_qp[n_x*i:(n_x*(i+1))]).T
        else:
            x_next = np.dot(A, np.matrix(x_qp[(n_x*(i-1)):(n_x*i)]).T)
            x_next += np.dot(B, np.matrix(x_qp[(nb_xf*n_x+nb_tot*3):(nb_xf*n_x+nb_tot*3+nb_contacts*3)]).T)
            x_next += g
            x_next += np.dot(A, xref[:, i:(i+1)]) - xref[:, (i+1):(i+2)]
            x_next -= np.matrix(x_qp[n_x*i:(n_x*(i+1))]).T
        nb_tot += nb_contacts
        print(x_next)
        if np.any(x_next > 0.001):
            print("ERROR: Dynamics of the system is broken.")
