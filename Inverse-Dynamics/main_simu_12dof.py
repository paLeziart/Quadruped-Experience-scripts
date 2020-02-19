# coding: utf8

import time
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pylab as plt

# import the controller class with its parameters
from TSID_Debug_controller_1LegRaised import controller, dt, q0, omega
import Safety_controller
import EmergencyStop_controller
import ForceMonitor
import robots_loader
from IPython import embed

########################################################################
#                        Parameters definition                         #
########################################################################

# Simulation parameters
N_SIMULATION = 4800  # number of time steps simulated

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
physicsClient = p.connect(p.DIRECT)
# p.GUI for graphical version
# p.DIRECT for non-graphical version

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
    baseState = p.getBasePositionAndOrientation(robotId)  # Position and orientation of the trunk
    baseVel = p.getBaseVelocity(robotId)  # Velocity of the trunk

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
    if(myController.error):
        print("Safety bounds reached. Switch to a safety controller")
        myController = mySafetyController

    # If the simulation time is too long, controller is switched to a zero torques controller
    time_error = time_error or (time.time()-time_start > 0.01)
    if (time_error):
        print("Computation time lasted to long. Switch to a zero torque control")
        myController = myEmergencyStop

    # Retrieve the joint torques from the appropriate controller
    """if i == 0:
        embed()"""
    jointTorques = myController.control(qmes12, vmes12, t, i, solo).reshape((12, 1))

    # Set control torque for all joints
    p.setJointMotorControlArray(robotId, revoluteJointIndices,
                                controlMode=p.TORQUE_CONTROL, forces=jointTorques)

    # Compute one step of simulation
    # p.stepSimulation()

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
        plt.plot((0.235 - 0.01205385) * np.ones((N_SIMULATION,)))
    plt.legend([l_str[i], "Reference"])

plt.figure()
plt.title("Trajectory of the CoM over time")
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(myController.com_pos_ref[:, i], "b", linewidth=2)
    plt.plot(myController.com_pos[:, i], "r", linewidth=2)
    plt.legend(["Ref pos along " + l_str[0], "Pos along " + l_str[0]])


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
