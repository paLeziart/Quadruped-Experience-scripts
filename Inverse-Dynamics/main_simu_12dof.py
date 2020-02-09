# coding: utf8

import time
import numpy as np
import pybullet as p
import pybullet_data
import pinocchio as pin
import matplotlib.pylab as plt

# import the controller class with its parameters
from TSID_MPC_controller import controller, dt, q0, omega
import Safety_controller
import EmergencyStop_controller
import ForceMonitor
import robots_loader

########################################################################
#                        Parameters definition                         #
########################################################################

# Simulation parameters
N_SIMULATION = 30000  # number of time steps simulated

t = 0.0  				# time

# Initialize the error for the simulation time
time_error = False

t_list = []

########################################################################
#                            Gepetto viewer                            #
########################################################################

solo = robots_loader.loadSolo(False)
solo.initDisplay(loadModel=True)
solo.viewer.gui.addFloor('world/floor')
solo.display(solo.q0)

########################################################################
#                              PyBullet                                #
########################################################################

# Start the client for PyBullet
physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version

# Load horizontal plane
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

# Set the gravity
p.setGravity(0, 0, -9.81)

# Load Quadruped robot
robotStartPos = [0, 0, 0.235]
robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
p.setAdditionalSearchPath("/opt/openrobots/share/example-robot-data/robots/solo_description/robots")
robotId = p.loadURDF("solo12.urdf", robotStartPos, robotStartOrientation)

# Disable default motor control for revolute joints
revoluteJointIndices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
p.setJointMotorControlArray(robotId, jointIndices=revoluteJointIndices, controlMode=p.VELOCITY_CONTROL,
                            targetVelocities=[0.0 for m in revoluteJointIndices],
                            forces=[0.0 for m in revoluteJointIndices])

# Initialize the robot in a specific configuration
straight_standing = np.array([[0, 0.8, -1.6, 0, 0.8, -1.6, 0, -0.8, 1.6, 0, -0.8, 1.6]]).transpose()
p.resetJointStatesMultiDof(robotId, revoluteJointIndices, straight_standing)  # q0[7:])

# Enable torque control for revolute joints
jointTorques = [0.0 for m in revoluteJointIndices]
p.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=p.TORQUE_CONTROL, forces=jointTorques)

# Fix the base in the world frame
# p.createConstraint(robotId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0.34])

# Set time step for the simulation
p.setTimeStep(dt)


########################################################################
#                             Simulator                                #
########################################################################

myController = controller(q0, omega, t)
mySafetyController = Safety_controller.controller_12dof()
myEmergencyStop = EmergencyStop_controller.controller_12dof()
myForceMonitor = ForceMonitor.ForceMonitor(p, robotId, planeId)

for i in range(N_SIMULATION):

    time_start = time.time()

    ####################################################################
    #                 Data collection from PyBullet                    #
    ####################################################################

    jointStates = p.getJointStates(robotId, revoluteJointIndices)  # State of all joints
    baseState = p.getBasePositionAndOrientation(robotId)
    baseVel = p.getBaseVelocity(robotId)

    # Joints configuration and velocity vector
    """qmes8 = np.vstack((np.array([baseState[0]]).T, np.array([baseState[1]]).T, np.array(
        [[jointStates[i_joint][0] for i_joint in range(len(jointStates))]]).T))
    vmes8 = np.vstack((np.zeros((6, 1)), np.array(
        [[jointStates[i_joint][1] for i_joint in range(len(jointStates))]]).T))"""

    # Conversion (from 8 to 12 DOF) for TSID computation
    """qmes12 = np.concatenate((qmes8[:7], np.matrix([0.]), qmes8[8:10], np.matrix(
        [0.]), qmes8[11:13], np.matrix([0.]), qmes8[14:16], np.matrix([0.]), qmes8[17:19]))
    vmes12 = np.concatenate((vmes8[:6], np.matrix([0.]), vmes8[7:9], np.matrix(
        [0.]), vmes8[10:12], np.matrix([0.]), vmes8[13:15], np.matrix([0.]), vmes8[16:18]))"""

    # Joints configuration and velocity vector for free-flyer + 12 dof
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
    if(myController.error):
        print("Safety bounds reached. Switch to a safety controller")
        myController = mySafetyController

    # If the simulation time is too long, controller is switched to a zero torques controller
    time_error = time_error or (time.time()-time_start > 0.01)
    if (time_error):
        print("Computation time lasted to long. Switch to a zero torque control")
        myController = myEmergencyStop

    # Retrieve the joint torques from the appropriate controller
    jointTorques = myController.control(qmes12, vmes12, t, solo).reshape((12, 1))

    # Set control torque for all joints
    p.setJointMotorControlArray(robotId, revoluteJointIndices,
                                controlMode=p.TORQUE_CONTROL, forces=jointTorques)

    # Compute one step of simulation
    p.stepSimulation()

    # Time incrementation
    t += dt

    time_spent = time.time() - time_start

    t_list.append(time_spent)

    myForceMonitor.display_contact_forces()

    if i % 250 == 0:
        a = 1

# Plot the tracking of the trajectories

"""plt.figure(1)
plt.plot(t_list, 'k+')

plt.show()"""
