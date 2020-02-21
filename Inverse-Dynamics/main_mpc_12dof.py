# coding: utf8

import time
import numpy as np
import pybullet as pyb
import pybullet_data
import matplotlib.pylab as plt

# import the controller class with its parameters
from TSID_Debug_controller_four_legs_fb_vel import controller, dt, q0, omega
import Safety_controller
import EmergencyStop_controller
import ForceMonitor

from IPython import embed

import Joystick
import MpcSolver
import settings
import FootstepPlanner
import FootTrajectoryGenerator
from mpc_functions import *
import utils

########################################################################
#                        Parameters definition                         #
########################################################################

# Simulation parameters
N_SIMULATION = 14800  # number of time steps simulated

t = 0.0  				# time

# Initialize the error for the simulation time
time_error = False

t_list = []

########################################################################
#                            Gepetto viewer                            #
########################################################################

solo = utils.init_viewer()

########################################################################
#                              PyBullet                                #
########################################################################

pyb_sim = utils.pybullet_simulator()

#########################################
#  Definition of parameters of the MPC  #
#########################################

# Position of the center of mass at the beginning
pos_CoM = np.array([[0.0], [0.0], [0.0]])

# Initial position of contacts (under shoulders)
settings.p_contacts = settings.shoulders.copy()

# Initial (x, y) positions of footholds
feet_target = np.array([[0.19, 0.19, -0.19, -0.19], [0.15005, -0.15005, 0.15005, -0.15005], [0.0, 0.0, 0.0, 0.0]])
p = np.array([[0.19, 0.19, -0.19, -0.19], [0.15005, -0.15005, 0.15005, -0.15005]])
goal_on_ground = np.array([[0.19, 0.19, -0.19, -0.19], [0.15005, -0.15005, 0.15005, -0.15005]])

# Initial (x, y) positions of footholds in the trunk frame
footholds_local = np.array([[0.19, 0.19, -0.19, -0.19], [0.15005, -0.15005, 0.15005, -0.15005]])

# Initial (x, y) target positions of footholds in the trunk frame
footholds_local_target = np.array([[0.19, 0.19, -0.19, -0.19], [0.15005, -0.15005, 0.15005, -0.15005]])

# Maximum height at which the robot should lift its feet during swing phase
# max_height_feet = 0.1

# Lock target positions of footholds before touchdown
t_lock_before_touchdown = 0.001

# Foot trajectory generator objects (one for each foot)
# ftgs = [Foot_trajectory_generator(max_height_feet, t_lock_before_touchdown) for i in range(4)]

# Number of loops
k_max_loop = 800

settings.h_ref = settings.qu_m[2, 0]

# Position of the trunk in the world frame
settings.q_w = (settings.qu_m).copy()

# Enable display with gepetto-gui
enable_gepetto_viewer = False

########################################################################
#                             Simulator                                #
########################################################################

myController = controller(q0, omega, t)
mySafetyController = Safety_controller.controller_12dof()
myEmergencyStop = EmergencyStop_controller.controller_12dof()
myForceMonitor = ForceMonitor.ForceMonitor(p, pyb_sim.robotId, pyb_sim.planeId)

for k in range(800):  # int(N_SIMULATION)):

    #####################
    #   MPC FUNCTIONS   #
    #####################

    # Run MPC once every 20 iterations of TSID

    if k == 0:
        settings.qu_m = np.array([[0.0, 0.0, 0.2027, 0.0, 0.0, 0.0]]).transpose()
        settings.vu_m = np.zeros((6, 1))
    elif k <= 828:
        RPY = utils.rotationMatrixToEulerAngles(myController.robot.framePosition(
            myController.invdyn.data(), myController.model.getFrameId("base_link")).rotation)
        """settings.qu_m[2] = myController.robot.framePosition(
                myController.invdyn.data(), myController.model.getFrameId("base_link")).translation[2, 0]"""
        settings.qu_m[2] = myController.robot.com(myController.invdyn.data())[2]
        # RPY[1] *= -1  # Pitch is inversed
        settings.qu_m[3:, 0] = RPY
        settings.qu_m[0:2, 0] = np.array([0.0, 0.0])
        settings.qu_m[5, 0] = 0.0
        settings.vu_m = myController.vtsid[:6, 0:1]
        if k == 10:
            settings.vu_m[0, 0] += 0.1
        # settings.vu_m[4] *= -1  # Pitch is inversed
    else:
        debug = 1

    ########################
    #  REFERENCE VELOCITY  #
    ########################

    # Create the joystick object
    if k == 0:
        joystick = Joystick.Joystick()

    # Update the reference velocity coming from the joystick
    joystick.update_v_ref(k)

    # Saving into settings
    settings.v_ref = joystick.v_ref

    # Get the reference velocity in global frame
    c, s = np.cos(settings.qu_m[5, 0]), np.sin(settings.qu_m[5, 0])
    R = np.array([[c, -s, 0., 0., 0., 0.], [s, c, 0., 0., 0., 0], [0., 0., 1.0, 0., 0., 0.],
                  [0., 0., 0., c, -s, 0.], [0., 0., 0., s, c, 0.], [0., 0., 0., 0., 0., 1.0]])
    settings.v_ref_world = np.dot(R, settings.v_ref)

    ######################
    #  CONTACT SEQUENCE  #
    ######################

    # Get contact sequence
    settings.t = settings.dt * k
    if k == 0:
        settings.S = footSequence(settings.t, settings.dt, settings.T_gait, settings.phases)
        settings.S = np.ones(settings.S.shape)
    """else:
        settings.S = np.vstack((settings.S[1:, :], settings.S[0:1, :]))"""

    ########################
    #  FOOTHOLDS LOCATION  #
    ########################

    # Create the objects during the first iteration then updating in the following iterations
    if k == 0:
        fstep_planner = FootstepPlanner.FootstepPlanner(0.03, settings.shoulders, settings.dt)
        ftraj_gen = FootTrajectoryGenerator.FootTrajectoryGenerator(settings.shoulders, settings.dt)
    else:
        ftraj_gen.update_frame(settings.vu_m)

    # Update desired location of footsteps using the footsteps planner
    fstep_planner.update_footsteps_mpc(settings.v_ref, settings.vu_m, settings.t_stance,
                                       settings.S, settings.T_gait, settings.qu_m[2, 0])

    # Updating quantities expressed in world frame
    fstep_planner.update_world_frame(settings.q_w)

    # Update 3D desired feet pos using the trajectory generator
    ftraj_gen.update_desired_feet_pos(fstep_planner.footsteps, settings.S,
                                      settings.dt, settings.T_gait - settings.t_stance, settings.q_w)

    # Get number of feet in contact with the ground for each step of the gait sequence
    settings.n_contacts = np.matrix(np.sum(settings.S, axis=1).astype(int)).T  # np.sum(settings.S, axis=1).astype(int)

    #########
    #  MPC  #
    #########

    # Create the MPC solver object
    if (k == 0):
        mpc = MpcSolver.MpcSolver(settings.dt, settings.S, k_max_loop)

    ##########################
    #  REFERENCE TRAJECTORY  #
    ##########################

    # Get the reference trajectory over the prediction horizon
    mpc.getRefStatesDuringTrajectory(settings)

    #####################
    #  SOLVER MATRICES  #
    #####################

    # Retrieve data from FootstepPlanner and FootTrajectoryGenerator
    mpc.retrieve_data(fstep_planner, ftraj_gen)

    # Create the constraints matrices used by the QP solver
    # Minimize x^T.P.x + x^T.q with constraints A.X == b and G.X <= h
    mpc.create_constraints_matrices(settings, solo, k)

    # Create the weights matrices used by the QP solver
    # P and q in the cost x^T.P.x + x^T.q
    if k == 0:  # Weight matrices are always the same
        mpc.create_weight_matrices(settings)

    #################
    #  CALL SOLVER  #
    #################

    # Create an initial guess and call the solver to solve the QP problem
    mpc.call_solver(settings)

    #####################
    #  RETRIEVE RESULT  #
    #####################

    # Extract relevant information from the output of the QP solver
    mpc.retrieve_result(settings)

    if k == 240:
        debug = 1

    #########################
    # UPDATE WORLD POSITION #
    #########################

    # Variation of position in world frame using the linear speed in local frame
    c_yaw, s_yaw = np.cos(settings.q_w[5, 0]), np.sin(settings.q_w[5, 0])
    R = np.array([[c_yaw, -s_yaw, 0], [s_yaw, c_yaw, 0], [0, 0, 1]])
    settings.q_w[0:3, 0:1] += np.dot(R, mpc.vu[0:3, 0:1] * settings.dt)

    # Variation of orientation in world frame using the angular speed in local frame
    settings.q_w[3:6, 0] += mpc.vu[3:6, 0] * settings.dt

    #####################
    #  GEPETTO VIEWER   #
    #####################

    if enable_gepetto_viewer:

        # Display non-locked target footholds with green spheres (gepetto gui)
        fstep_planner.update_viewer(solo.viewer, (k == 0))

        # Display locked target footholds with red spheres (gepetto gui)
        # Display desired 3D position of feet with magenta spheres (gepetto gui)
        ftraj_gen.update_viewer(solo.viewer, (k == 0))

        # Display reference trajectory, predicted trajectory, desired contact forces, current velocity
        mpc.update_viewer(solo.viewer, (k == 0), settings)

        qu_pinocchio = solo.q0
        qu_pinocchio[0:3, 0:1] = settings.q_w[0:3, 0:1]
        # TODO: Check why orientation of q_w and qu are different
        #qu_pinocchio[3:7, 0:1] = getQuaternion(settings.q_w[3:6, 0:1])
        qu_pinocchio[3:7, 0:1] = utils.getQuaternion(mpc.qu[3:6, 0:1])

        # Refresh the gepetto viewer display
        solo.display(qu_pinocchio)
        # solo.viewer.gui.refresh()

    # Get measured position and velocity after one time step
    # settings.qu_m, settings.vu_m = low_pass_robot(qu, vu)
    settings.qu_m[[2, 3, 4]] = mpc.qu[[2, 3, 4]]  # coordinate in x, y, yaw is always 0 in local frame
    settings.vu_m = mpc.vu

    # print(mpc.f_applied)
    #print("END OF MPC ITERATION")

    for i in range(1):

        time_start = time.time()

        ####################################################################
        #                 Data collection from PyBullet                    #
        ####################################################################

        jointStates = pyb.getJointStates(pyb_sim.robotId, pyb_sim.revoluteJointIndices)  # State of all joints
        baseState = pyb.getBasePositionAndOrientation(pyb_sim.robotId)  # Position and orientation of the trunk
        baseVel = pyb.getBaseVelocity(pyb_sim.robotId)  # Velocity of the trunk

        # Joints configuration and velocity vector for free-flyer + 12 actuators
        qmes12 = np.vstack((np.array([baseState[0]]).T, np.array([baseState[1]]).T,
                            np.array([[jointStates[i_joint][0] for i_joint in range(len(jointStates))]]).T))
        vmes12 = np.vstack((np.array([baseVel[0]]).T, np.array([baseVel[1]]).T,
                            np.array([[jointStates[i_joint][1] for i_joint in range(len(jointStates))]]).T))

        ####################################################################
        #                Select the appropriate controller 				   #
        #                               &								   #
        #               Load the joint torques into the robot			   #
        ####################################################################

        # If the limit bounds are reached, controller is switched to a pure derivative controller
        """if(myController.error):
            print("Safety bounds reached. Switch to a safety controller")
            myController = mySafetyController"""

        # If the simulation time is too long, controller is switched to a zero torques controller
        """time_error = time_error or (time.time()-time_start > 0.01)
        if (time_error):
            print("Computation time lasted to long. Switch to a zero torque control")
            myController = myEmergencyStop"""

        # Retrieve the joint torques from the appropriate controller
        jointTorques = myController.control(qmes12, vmes12, t, i+k, solo, mpc).reshape((12, 1))

        # Set control torque for all joints
        pyb.setJointMotorControlArray(pyb_sim.robotId, pyb_sim.revoluteJointIndices,
                                      controlMode=pyb.TORQUE_CONTROL, forces=jointTorques)

        # Compute one step of simulation
        # pyb.stepSimulation()

        # Time incrementation
        t += dt

        # Time spent to run this iteration of the loop
        time_spent = time.time() - time_start

        # Logging the time spent
        t_list.append(time_spent)

        # Refresh force monitoring for PyBullet
        # myForceMonitor.display_contact_forces()
        # time.sleep(0.001)

# Plot the time spent to run each iteration of the loop

plt.figure(1)
plt.title("Trajectory of the front right foot over time")
l_str = ["X", "Y", "Z"]
for i in range(3):
    plt.subplot(3, 1, 1*i+1)
    plt.plot(myController.f_pos_ref[1, :, i])
    plt.plot(myController.f_pos[1, :, i])
    plt.legend(["Ref pos along " + l_str[i], "Pos along " + l_str[i]])

plt.figure()
plt.title("Velocity of the front right foot over time")
l_str = ["X", "Y", "Z"]
for i in range(3):
    plt.subplot(3, 1, 1*i+1)
    plt.plot(myController.f_vel_ref[1, :, i])
    plt.plot(myController.f_vel[1, :, i])
    plt.legend(["Ref vel along " + l_str[i], "Vel along " + l_str[i]])
    """plt.subplot(3, 3, 3*i+3)
    plt.plot(myController.f_acc_ref[1, :, i])
    plt.plot(myController.f_acc[1, :, i])
    plt.legend(["Ref acc along " + l_str[i], "Acc along " + l_str[i]])"""

plt.figure()
l_str = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(myController.b_pos[:, i])
    if i < 2:
        plt.plot(np.zeros((N_SIMULATION,)))
    else:
        plt.plot((0.2027) * np.ones((N_SIMULATION,)))
    plt.legend([l_str[i], "Reference"])

if hasattr(myController, 'com_pos_ref'):
    plt.figure()
    plt.title("Trajectory of the CoM over time")
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(myController.com_pos_ref[:, i], "b", linewidth=2)
        plt.plot(myController.com_pos[:, i], "r", linewidth=2)
        plt.legend(["COM Ref pos along " + l_str[0], "Pos along " + l_str[0]])


plt.show()

plt.figure(9)
plt.plot(t_list, 'k+')
plt.show()

quit()


plt.figure(2)
c = ["r", "g", "b", "k"]
for i in range(3):
    plt.subplot(3, 1, i+1)
    for k in range(1, 2):
        plt.plot(myController.p_feet[k, :, i], color=c[k], linewidth=2)
        plt.plot(myController.p_contacts[k, :, i], linestyle='dashed', linewidth=2)
        plt.plot(myController.p_tracking[k, :, i], linestyle='dotted', linewidth=2)
        # plt.plot(myController.p_traj_gen[k, :, i], color="rebeccapurple", linewidth=2)
        plt.legend(["Position réelle", "Position du contact", "Position du tracking"])

        """cpt = 0
        while cpt < (N_SIMULATION-10):
            plt.plot(np.arange(cpt, cpt+300, 1),
                     myController.p_contacts[k, cpt:(cpt+300), i], color=c[k], linestyle='dashed')
            plt.plot(np.arange(cpt+300, cpt+600, 1),
                     myController.p_tracking[k, (cpt+300):(cpt+600), i], color=c[k], linestyle="dotted")
            cpt += 600"""
plt.show()

plt.figure(3)
c = ["r", "g", "b", "k"]
for i in range(3):
    plt.subplot(3, 3, 3*i+1)
    for k in range(1, 2):
        plt.plot(myController.p_traj_gen[k, :, i], color=c[k], linewidth=2)
        # plt.legend(["FL", "FR", "HL", "HR"])
    plt.subplot(3, 3, 3*i+2)
    for k in range(1, 2):
        plt.plot(myController.p_vel[k, :, i], color=c[k], linewidth=2)
    plt.subplot(3, 3, 3*i+3)
    for k in range(1, 2):
        plt.plot(myController.p_acc[k, :, i], color=c[k], linewidth=2)
plt.show()

plt.figure(4)
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(myController.p_base[:, i], linestyle='dotted', linewidth=2)
plt.show()
