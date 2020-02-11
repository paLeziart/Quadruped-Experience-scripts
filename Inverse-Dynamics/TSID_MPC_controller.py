# coding: utf8


########################################################################
#                                                                      #
#          Control law : tau = P(q*-q^) + D(v*-v^) + tau_ff            #
#                                                                      #
########################################################################

import pinocchio as pin
import numpy as np
import numpy.matlib as matlib
import tsid

pin.switchToNumpyMatrix()


########################################################################
#            Class for a PD with feed-forward Controller               #
########################################################################

class controller:

    def __init__(self, q0, omega, t):

        self.omega = omega
        self.qdes = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                               0.0, 0.8, -1.6, 0, 0.8, -1.6,
                               0, -0.8, 1.6, 0, -0.8, 1.6]]).transpose()  # q0.copy()
        self.vdes = np.zeros((18, 1))
        self.ades = np.zeros((18, 1))
        self.error = False
        self.verbose = False

        # List with the names of all feet frames
        self.foot_frames = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']

        # Constraining the contacts
        mu = 0.3  				# friction coefficient
        fMin = 1.0				# minimum normal force
        fMax = 100.0  			# maximum normal force
        contactNormal = np.matrix([0., 0., 1.]).T  # direction of the normal to the contact surface

        # Coefficients of the posture task
        kp_posture = 10.0		# proportionnal gain of the posture task
        w_posture = 1.0			# weight of the posture task

        # Coefficients of the contact tasks
        kp_contact = 20000.0		# proportionnal gain for the contacts
        self.w_forceRef = 1e-5		# weight of the forces regularization

        # Coefficients of the foot tracking tasks
        kp_foot = 20000.0
        self.w_foot = 1000.0

        self.pair = 0  # Which pair of feet is touching the ground
        self.init = False  # Flag for first iteration
        self.dz = 0.0003  # Vertical displacement for each iteration
        self.velz = 0.3  # Vertical velocity for each iteration

        ########################################################################
        #             Definition of the Model and TSID problem                 #
        ########################################################################

        # Set the paths where the urdf and srdf file of the robot are registered

        modelPath = "/opt/openrobots/share/example-robot-data/robots"
        urdf = modelPath + "/solo_description/robots/solo12.urdf"
        srdf = modelPath + "/solo_description/srdf/solo.srdf"
        vector = pin.StdVec_StdString()
        vector.extend(item for item in modelPath)

        # Create the robot wrapper from the urdf model (which has no free flyer) and add a free flyer
        self.robot = tsid.RobotWrapper(urdf, vector, pin.JointModelFreeFlyer(), False)
        self.model = self.robot.model()

        # Creation of the Invverse Dynamics HQP problem using the robot
        # accelerations (base + joints) and the contact forces
        self.invdyn = tsid.InverseDynamicsFormulationAccForce("tsid", self.robot, False)
        # Compute the problem data with a solver based on EiQuadProg
        self.invdyn.computeProblemData(t, self.qdes, self.vdes)

        #####################
        # LEGS POSTURE TASK #
        #####################

        # Task definition (creating the task object)
        self.postureTask = tsid.TaskJointPosture("task-posture", self.robot)
        self.postureTask.setKp(kp_posture * matlib.ones(self.robot.nv-6).T)  # Proportional gain
        self.postureTask.setKd(2.0 * np.sqrt(kp_posture) * matlib.ones(self.robot.nv-6).T)  # Derivative gain
        # Add the task to the HQP with weight = w_posture, priority level = 1 (not real constraint)
        # and a transition duration = 0.0
        self.invdyn.addMotionTask(self.postureTask, w_posture, 1, 0.0)

        # TSID Trajectory (creating the trajectory object and linking it to the task)
        pin.loadReferenceConfigurations(self.model, srdf, False)
        self.q_ref = self.model.referenceConfigurations['straight_standing']
        self.q_ref[2, 0] = 0  # Discard height
        self.trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", self.q_ref[7:])
        # Set the trajectory as reference of the posture task
        self.samplePosture = self.trajPosture.computeNext()
        self.postureTask.setReference(self.samplePosture)

        # Start TSID with the robot in the reference configuration
        self.qdes = self.q_ref

        #########################
        # CONTACTS AND TRACKING #
        #########################

        self.contacts = 4*[None]  # List to store the rigid contact tasks
        self.feetTask = 4*[None]  # List to store the feet tracking tasks
        self.feetGoal = 4*[None]  # List to store the tracking goals
        self.feetTraj = 4*[None]  # List to store the trajectory objects

        for i, name in enumerate(self.foot_frames):

            ##########################
            # RIGID CONTACTS OBJECTS #
            ##########################

            self.contacts[i] = tsid.ContactPoint(name, self.robot, name, contactNormal, mu, fMin, fMax)
            self.contacts[i].setKp(kp_contact * matlib.ones(3).T)
            self.contacts[i].setKd(2.0 * np.sqrt(kp_contact) * matlib.ones(3).T)
            self.contacts[i].useLocalFrame(False)
            H_ref = self.robot.framePosition(self.invdyn.data(), self.model.getFrameId(name))
            self.contacts[i].setReference(H_ref)

            ############################
            # REFERENCE CONTACT FORCES #
            ############################

            """
            self.contacts[i].setForceReference(np.matrix([0.0, 0.0, 29.21 * 0.35]).T)  # 2.2 * 9.81 * 0.25
            self.contacts[i].setRegularizationTaskWeightVector(np.matrix([1., 1., 1.]).T)
            """

            # Adding the rigid contact after the reference contact force has been set
            self.invdyn.addRigidContact(self.contacts[i], self.w_forceRef)

            #######################
            # FEET TRACKING TASKS #
            #######################

            self.feetTask[i] = tsid.TaskSE3Equality(name+"_track", self.robot, name)
            mask = np.matrix([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).T
            self.feetTask[i].setKp(kp_foot * mask)
            self.feetTask[i].setKd(2.0 * np.sqrt(kp_foot) * mask)  #  matlib.ones(6).T)
            self.feetTask[i].setMask(mask)
            self.feetTask[i].useLocalFrame(False)

            # Add the task to the HQP with weight = w_foot, priority level = 1 (not real constraint)
            #  and a transition duration = 0.0
            # self.invdyn.addMotionTask(self.feetTask[i], self.w_foot, 1, 0.0)

            # Get the starting position/orientation of the foot frame
            self.feetGoal[i] = self.robot.framePosition(self.invdyn.data(), self.model.getFrameId(name))
            self.feetGoal[i].translation = np.matrix(
                [self.feetGoal[i].translation[0, 0], self.feetGoal[i].translation[1, 0], -0.223]).T
            self.feetTraj[i] = tsid.TrajectorySE3Constant(name+"_track", self.feetGoal[i])
            # Set the trajectory as reference of the tracking task
            self.sampleFoot = self.feetTraj[i].computeNext()
            self.feetTask[i].setReference(self.sampleFoot)

        ##########
        # SOLVER #
        ##########

        # Use EiquadprogFast solver
        self.solver = tsid.SolverHQuadProgFast("qp solver")

        # Resize the solver to fit the number of variables, equality and inequality constraints
        self.solver.resize(self.invdyn.nVar, self.invdyn.nEq, self.invdyn.nIn)

    ####################################################################
    #                Modification foot tracking method                 #
    ####################################################################

    def move_vertical(self, i_foot, dz, end):

        # Adding a vertical offset to the current goal
        self.feetGoal[i_foot].translation = np.matrix(
            [self.feetGoal[i_foot].translation[0, 0],
             self.feetGoal[i_foot].translation[1, 0],
             self.feetGoal[i_foot].translation[2, 0] + dz]).T
        self.feetTraj[i_foot] = tsid.TrajectorySE3Constant("traj_FR_foot", self.feetGoal[i_foot])
        self.sampleFoot = self.feetTraj[i_foot].computeNext()
        self.sampleFoot.vel(np.array([[0.0, 0.0, dz * 1000, 0.0, 0.0, 0.0]]).transpose())

        # If the foot is going to enter stance phase then the desired velocity is 0
        if end:
            self.sampleFoot.vel(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).transpose())

        # Setting the new reference
        self.feetTask[i_foot].setReference(self.sampleFoot)

        return 0

    ####################################################################
    #                      Torque Control method                       #
    ####################################################################
    def control(self, qmes12, vmes12, t, solo):

        # Round the time with numpy to avoid numerical effects like 0.009999... instead of 0.001
        t = np.round(t, decimals=3)

        if self.verbose:
            print("## Time: ", t)

        # Set TSID state to the state of PyBullet simulation
        self.qdes[:3] = np.zeros((3, 1))  # Discard x and y drift and height position
        self.vdes[0:3] = np.zeros((3, 1))  # Discard horizontal and vertical velocities

        # Handling contacts and feet tracking to perform a walking trot with a period of 0.6 s
        if self.init:
            if np.abs(t % 0.3) < 1e-4:  # If time is a multiple of 0.3 then we enter a four contacts phase
                if self.pair == 0:
                    for i_foot in [0, 3]:
                        self.contacts[i_foot].setReference(self.feetGoal[i_foot])
                        self.invdyn.addRigidContact(self.contacts[i_foot], self.w_forceRef)
                        self.invdyn.removeTask(self.foot_frames[i_foot]+"_track", 0.0)
                    self.pair = 1
                else:
                    for i_foot in [1, 2]:
                        self.contacts[i_foot].setReference(self.feetGoal[i_foot])
                        self.invdyn.addRigidContact(self.contacts[i_foot], self.w_forceRef)
                        self.invdyn.removeTask(self.foot_frames[i_foot]+"_track", 0.0)
                    self.pair = 0
            elif np.abs((t % 0.3)-0.001) < 1e-4:  # Feet leaving the ground after a four contacts phase
                if self.pair == 0:
                    for i_foot in [0, 3]:
                        self.invdyn.removeRigidContact(self.foot_frames[i_foot], 0.0)
                        self.invdyn.addMotionTask(self.feetTask[i_foot], self.w_foot, 1, 0.0)
                else:
                    for i_foot in [1, 2]:
                        self.invdyn.removeRigidContact(self.foot_frames[i_foot], 0.0)
                        self.invdyn.addMotionTask(self.feetTask[i_foot], self.w_foot, 1, 0.0)

            if (t % 0.3) > 0 and (t % 0.3) <= 0.15:  # Feet in swing phase moving upwards during 0.15 s
                if self.pair == 0:
                    for i_foot in [0, 3]:
                        self.move_vertical(i_foot, 0.0003, False)
                else:
                    for i_foot in [1, 2]:
                        self.move_vertical(i_foot, 0.0003, False)
            elif (t % 0.3) > 0.15:  # Feet in swing phase moving downwards during 0.15 s
                if self.pair == 0:
                    for i_foot in [0, 3]:
                        self.move_vertical(i_foot, -0.0003, (t == 0.299))
                else:
                    for i_foot in [1, 2]:
                        self.move_vertical(i_foot, -0.0003, (t == 0.299))
        else:
            self.init = True

        if np.abs(t % 0.1) < 0.001:
            debug = 1

        # Old code for smooth transition between the current position and the current target position
        """# Target SE3 for the foot (position/orientation) in world frame
            goal_x = (0.19 + 0.05 * np.sin(2*3.1415*1*t)) * np.min((1.0, t)) + \
                self.FRfootTask.position[0, 0] * (1.0 - np.min((1.0, t)))
            goal_y = -0.15 * np.min((1.0, t)) + self.FRfootTask.position[1, 0] * (1.0 - np.min((1.0, t)))
            goal_z = (0.1 + 0.03 * np.sin(2*3.1415*1*t)) * np.min((1.0, t)) + \
                self.FRfootTask.position[2, 0] * (1 - np.min((1.0, t)))

            # Set the updated target SE3 as reference for the tracking task
            self.sampleFoot.pos(np.matrix([goal_x, goal_y, goal_z,
                                           1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).T)
            self.sampleFoot.vel(np.matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T)
            self.sampleFoot.acc(np.matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T)"""

        # Resolution of the HQP problem
        HQPData = self.invdyn.computeProblemData(t, self.qdes, self.vdes)
        self.sol = self.solver.solve(HQPData)

        # Torques, accelerations, velocities and configuration computation
        tau_ff = self.invdyn.getActuatorForces(self.sol)
        self.ades = self.invdyn.getAccelerations(self.sol)
        self.vdes += self.ades * dt
        self.qdes = pin.integrate(self.model, self.qdes, self.vdes * dt)

        # Get contact forces for debug purpose
        if self.verbose:
            ctc_forces = self.invdyn.getContactForces(self.sol)
            nb_feet = int(ctc_forces.shape[0] / 3)
            for i_foot in range(nb_feet):
                print("Contact forces foot ", i_foot, ": ", ctc_forces[(i_foot*3):(i_foot*3+3), 0].transpose())
            for i, name in enumerate(['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']):
                pos_foot = self.robot.framePosition(self.invdyn.data(), self.model.getFrameId(name))
                print("Foot ", i, "at position ", pos_foot.translation.transpose())
                print(i, " desired at position ", self.feetGoal[i].translation.transpose())

            # Display target 3D positions of footholds with green spheres (gepetto gui)
            rgbt = [0.0, 1.0, 0.0, 0.5]
            for i in range(4):
                if (t == 0):
                    solo.viewer.gui.addSphere("world/sphere"+str(i)+"_target", .02, rgbt)  # .1 is the radius
                solo.viewer.gui.applyConfiguration(
                    "world/sphere"+str(i)+"_target", (self.feetGoal[i].translation[0, 0],
                                                      self.feetGoal[i].translation[1, 0],
                                                      self.feetGoal[i].translation[2, 0], 1., 0., 0., 0.))

            # Display current 3D positions of footholds with magenta spheres (gepetto gui)
            rgbt = [1.0, 0.0, 1.0, 0.5]
            for i in range(4):
                if (t == 0):
                    solo.viewer.gui.addSphere("world/sphere"+str(i)+"_pos", .02, rgbt)  # .1 is the radius
                pos_foot = self.robot.framePosition(self.invdyn.data(), self.model.getFrameId(self.foot_frames[i]))
                solo.viewer.gui.applyConfiguration(
                    "world/sphere"+str(i)+"_pos", (pos_foot.translation[0, 0],
                                                   pos_foot.translation[1, 0],
                                                   pos_foot.translation[2, 0], 1., 0., 0., 0.))

            solo.viewer.gui.refresh()

            # Refresh gepetto gui with TSID desired joint position
            solo.display(self.qdes)

        # Torque PD controller
        P = 50  # 50
        D = 1  #  0.2
        torques12 = P * (self.qdes[7:] - qmes12[7:]) + D * (self.vdes[6:] - vmes12[6:]) + tau_ff

        # Saturation to limit the maximal torque
        t_max = 2.5
        tau = np.clip(torques12, -t_max, t_max)  # faster than np.maximum(a_min, np.minimum(a, a_max))

        # self.error = self.error or (self.sol.status != 0) or (qmes12[8] < -np.pi/2) or (
        #              qmes12[11] < -np.pi/2) or (qmes12[14] < -np.pi/2) or (qmes12[17] < -np.pi/2) or (
        #              qmes12[8] > np.pi/2) or (qmes12[11] > np.pi/2) or (qmes12[14] > np.pi/2) or (
        #              qmes12[17] > np.pi/2)

        return tau.flatten()

# Parameters for the controller


dt = 0.001				# controller time step

q0 = np.zeros((19, 1))  # initial configuration

omega = 1.0				# sinus pulsation
