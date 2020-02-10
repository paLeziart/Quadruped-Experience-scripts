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
        self.qdes = q0.copy()
        self.vdes = np.zeros((18, 1))
        self.ades = np.zeros((18, 1))
        self.error = False

        kp_posture = 10.0		# proportionnal gain of the posture task
        w_posture = 0.0			# weight of the posture task

        kp_lock = 0.0			# proportionnal gain of the lock task
        w_lock = 0.0			# weight of the lock task

        # For the contacts
        mu = 0.3  				# friction coefficient
        fMin = 1.0				# minimum normal force
        fMax = 100.0  			# maximum normal force

        w_forceRef = 1e-3		# weight of the forces regularization
        self.w_forceRef = w_forceRef
        kp_contact = 0.0		# proportionnal gain for the contacts

        self.foot_frames = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']  # tab with all the foot frames names
        contactNormal = np.matrix([0., 0., 1.]).T  # direction of the normal to the contact surface

        kp_com = 0.0
        w_com = 0.0

        kp_foot = 100.0
        w_foot = 1000.0

        self.pair = 0
        self.init = False
        self.dz = 0.0003
        self.velz = 0.3

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
        # Get the initial data
        self.data = self.invdyn.data()

        #####################
        # LEGS POSTURE TASK #
        #####################

        # Task definition (creating the task object)
        self.postureTask = tsid.TaskJointPosture("task-posture", self.robot)
        self.postureTask.setKp(kp_posture * matlib.ones(self.robot.nv-6).T)  # Proportional gain
        self.postureTask.setKd(2.0 * np.sqrt(kp_posture) * matlib.ones(self.robot.nv-6).T)  # Derivative gain
        # self.postureTask.mask(np.matrix([[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]]).T)
        # Add the task to the HQP with weight = w_posture, priority level = 0 (as real constraint) and a transition duration = 0.0
        self.invdyn.addMotionTask(self.postureTask, w_posture, 1, 0.0)

        # TSID Trajectory (creating the trajectory object and linking it to the task)
        pin.loadReferenceConfigurations(self.model, srdf, False)
        self.q_ref = self.model.referenceConfigurations['straight_standing']
        self.trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", self.q_ref[7:])
        # Set the trajectory as reference for the posture task
        self.samplePosture = self.trajPosture.computeNext()
        self.postureTask.setReference(self.samplePosture)

        # Start in reference configuration
        self.qdes = self.q_ref

        ######################
        # TRUNK POSTURE TASK #
        ######################

        """
        # Task definition (creating the task object)
        self.comTask = tsid.TaskComEquality("task-com", self.robot)
        mask = np.matrix([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T
        self.comTask.setKp(kp_com * mask)
        self.comTask.setKd(2.0 * np.sqrt(kp_com) * mask)
        self.invdyn.addMotionTask(self.comTask, w_com, 1, 0.0)

        # TSID Trajectory (creating the trajectory object and linking it to the task)
        self.com_ref = self.robot.com(self.data)
        self.trajCom = tsid.TrajectoryEuclidianConstant("traj_com", self.com_ref)
        self.sampleCom = self.trajCom.computeNext()
        self.sampleCom.pos(np.matrix([0.0, 0.0, 0.0]).T)
        self.sampleCom.vel(np.matrix([0.0, 0.0, 0.0]).T)
        self.sampleCom.acc(np.matrix([0.0, 0.0, 0.0]).T)
        self.comTask.setReference(self.sampleCom)
        """

        # Task definition (creating the task object)
        """self.comTaskSe3 = tsid.TaskSE3Equality("task-com-se3", self.robot, 'base_link')
        maskKp = np.matrix([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).T
        maskKd = np.matrix([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).T

        self.comTaskSe3.setKp(kp_com * maskKp)
        self.comTaskSe3.setKd(2.0 * np.sqrt(kp_com) * maskKd)
        self.comTaskSe3.useLocalFrame(False)
        self.invdyn.addMotionTask(self.comTaskSe3, w_com, 1, 0.0)

        # TSID Trajectory (creating the trajectory object and linking it to the task)
        self.com_ref = self.robot.framePosition(self.data, self.model.getFrameId('base_link'))
        self.trajCom = tsid.TrajectorySE3Constant("traj_base_link", self.com_ref)
        self.sampleCom = self.trajCom.computeNext()
        self.sampleCom.pos(np.matrix([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).T)
        self.sampleCom.vel(np.matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T)
        self.sampleCom.acc(np.matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T)
        self.comTaskSe3.setReference(self.sampleCom)"""

        ########################
        # CONTACTS CONSTRAINTS #
        ########################

        self.contacts = 4*[None]
        self.feetTask = 4*[None]
        self.feetGoal = 4*[None]
        self.feetTraj = 4*[None]
        for i, name in enumerate(self.foot_frames):
            self.contacts[i] = tsid.ContactPoint(name, self.robot, name, contactNormal, mu, fMin, fMax)
            self.contacts[i].setKp(kp_contact * matlib.ones(3).T)
            self.contacts[i].setKd(2.0 * np.sqrt(kp_contact) * matlib.ones(3).T)
            H_ref = self.robot.framePosition(self.data, self.model.getFrameId(name))
            self.contacts[i].setReference(H_ref)
            self.contacts[i].useLocalFrame(False)

            ############################
            # REFERENCE CONTACT FORCES #
            ############################

            """if i == 3:
                self.contacts[i].setForceReference(np.matrix([0.0, 0.0, 29.21 * 0.35]).T)  # 2.2 * 9.81 * 0.25
                self.contacts[i].setRegularizationTaskWeightVector(np.matrix([1., 1., 1.]).T)"""

            self.invdyn.addRigidContact(self.contacts[i], w_forceRef)

            ##########################
            # CONTACT TRACKING TASKS #
            ##########################

            self.feetTask[i] = tsid.TaskSE3Equality(name+"_track", self.robot, name)
            mask = np.matrix([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).T
            self.feetTask[i].setKp(kp_foot * mask)  # matlib.ones(6).T)
            self.feetTask[i].setKd(2.0 * np.sqrt(kp_foot) * mask)  # matlib.ones(6).T)
            # Add the task to the HQP with weight = w_foot, priority level = 0 (as real constraint) and a transition duration = 0.0
            self.feetTask[i].useLocalFrame(False)
            self.invdyn.addMotionTask(self.feetTask[i], w_foot, 1, 0.0)

            # Get the current position/orientation of the foot frame
            self.feetGoal[i] = self.robot.framePosition(self.data, self.model.getFrameId(name))
            self.feetGoal[i].translation = np.matrix(
                [self.feetGoal[i].translation[0, 0], self.feetGoal[i].translation[1, 0], -0.223]).T
            self.feetTraj[i] = tsid.TrajectorySE3Constant(name+"_track", self.feetGoal[i])
            self.sampleFoot = self.feetTraj[i].computeNext()
            self.feetTask[i].setReference(self.sampleFoot)

        ####################
        # FOOT MOTION TASK #
        ####################

        """# Task definition (creating the task object)
        self.FRfootTask = tsid.TaskSE3Equality("FR-foot-placement", self.robot, 'FR_FOOT')

        # ignore rotation for contact points
        mask = np.matrix([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).T
        self.FRfootTask.setKp(kp_foot * mask)  # matlib.ones(6).T)
        self.FRfootTask.setKd(2.0 * np.sqrt(kp_foot) * mask)  # matlib.ones(6).T)
        # set a mask allowing only the transation upon x and z-axis
        # self.FRfootTask.setMask(np.matrix([[1, 0, 1, 0, 0, 0]]).T)
        self.FRfootTask.useLocalFrame(False)
        # Add the task to the HQP with weight = w_foot, priority level = 0 (as real constraint) and a transition duration = 0.0
        self.invdyn.addMotionTask(self.FRfootTask, w_foot, 1, 0.0)

        # TSID Trajectory (creating the trajectory object and linking it to the task)
        # pin.forwardKinematics(self.model, self.data, self.qdes)
        # pin.updateFramePlacements(self.model, self.data)

        # Get the current position/orientation of the foot frame
        self.FR_foot_ref = self.robot.framePosition(self.data, self.model.getFrameId('FR_FOOT'))
        # Set the goal 5 cm above the starting location of the foot
        FRgoalz = -0.223  # -0.223 # self.FR_foot_ref.translation[2, 0]
        self.FR_foot_goal = self.FR_foot_ref.copy()
        self.FR_foot_goal.translation = np.matrix(
            [self.FR_foot_ref.translation[0, 0], self.FR_foot_ref.translation[1, 0], FRgoalz]).T

        # Set the trajectory as reference for the foot positionning task
        self.trajFRfoot = tsid.TrajectorySE3Constant("traj_FR_foot", self.FR_foot_goal)
        self.sampleFoot = self.trajFRfoot.computeNext()
        self.FRfootTask.setReference(self.sampleFoot)"""

        ##########
        # SOLVER #
        ##########

        # Initialization of the solver

        # Use EiquadprogFast solver
        self.solver = tsid.SolverHQuadProgFast("qp solver")
        # Resize the solver to fit the number of variables, equality and inequality constraints
        self.solver.resize(self.invdyn.nVar, self.invdyn.nEq, self.invdyn.nIn)

    ####################################################################
    #                Modification foot tracking method                 #
    ####################################################################

    def move_vertical(self, i_foot, dz, end):
        self.feetGoal[i_foot].translation = np.matrix(
            [self.feetGoal[i_foot].translation[0, 0],
             self.feetGoal[i_foot].translation[1, 0],
             self.feetGoal[i_foot].translation[2, 0] + dz]).T
        self.feetTraj[i_foot] = tsid.TrajectorySE3Constant("traj_FR_foot", self.feetGoal[i_foot])
        self.sampleFoot = self.feetTraj[i_foot].computeNext()
        self.sampleFoot.vel(np.array([[0.0, 0.0, dz * 1000, 0.0, 0.0, 0.0]]).transpose())
        if end:
            self.sampleFoot.vel(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).transpose())
        self.feetTask[i_foot].setReference(self.sampleFoot)

        return 0

    ####################################################################
    #                      Torque Control method                       #
    ####################################################################
    def control(self, qmes12, vmes12, t, solo):
        t = np.round(t, decimals=3)

        print("## Time: ", t)
        # Set TSID state to the state of PyBullet simulation
        self.qdes[:2] = np.zeros((2, 1))  # Discard x and y drift
        self.qdes[2] = 0.0  # Discard height position
        #self.vdes[0:3] = np.zeros((3, 1))
        """self.qdes[2:7] = qmes12[2:7]  # Keep height and orientation
        self.vdes[:6] = vmes12[:6]
        self.vdes = vmes12
        self.qdes = qmes12
        self.qdes[:2] = np.zeros((2, 1))  # Discard x and y drift"""

        # Update frame placements
        pin.forwardKinematics(self.model, self.data, self.qdes, self.vdes)
        pin.updateFramePlacements(self.model, self.data)

        # Sinusoidal reference for the contact force task
        #self.contacts[3].setForceReference(np.matrix([0.0, 0.0, 4.0 + 3.0 * np.sin(2*3.1415*0.2*t)]).T)
        #self.contacts[2].setForceReference(np.matrix([0.0, 0.0, 4.0 - 3.0 * np.sin(2*3.1415*0.2*t)]).T)

        if self.init:
            if np.abs(t % 0.3) < 1e-4:
                if self.pair == 0:
                    for i_foot in [0, 3]:
                        self.contacts[i_foot].setReference(self.feetGoal[i_foot])
                        self.invdyn.addRigidContact(self.contacts[i_foot], self.w_forceRef)
                    self.pair = 1
                else:
                    for i_foot in [1, 2]:
                        self.contacts[i_foot].setReference(self.feetGoal[i_foot])
                        self.invdyn.addRigidContact(self.contacts[i_foot], self.w_forceRef)
                    self.pair = 0
            elif np.abs((t % 0.3)-0.001) < 1e-4:
                if self.pair == 0:
                    for i_foot in [0, 3]:
                        self.invdyn.removeRigidContact(self.foot_frames[i_foot], 0.0)
                else:
                    for i_foot in [1, 2]:
                        self.invdyn.removeRigidContact(self.foot_frames[i_foot], 0.0)

            if (t % 0.3) > 0 and (t % 0.3) <= 0.15:
                if self.pair == 0:
                    for i_foot in [0, 3]:
                        self.move_vertical(i_foot, 0.0003, False)
                else:
                    for i_foot in [1, 2]:
                        self.move_vertical(i_foot, 0.0003, False)
            elif (t % 0.3) > 0.15:
                if self.pair == 0:
                    for i_foot in [0, 3]:
                        self.move_vertical(i_foot, -0.0003, (t == 0.299))
                else:
                    for i_foot in [1, 2]:
                        self.move_vertical(i_foot, -0.0003, (t == 0.299))
        else:
            self.init = True

        if np.abs(t % 0.1) < 0.001:
            trigger = 1

        """if np.abs(t - 0.2) < 0.0001:
            for i_foot in [1, 2]:
                self.invdyn.removeRigidContact(self.foot_frames[i_foot], 0.0)

        if np.abs(t - 0.5) < 0.0001:
            for i_foot in [1, 2]:
                #H_ref = self.robot.framePosition(self.data, self.model.getFrameId(self.foot_frames[i_foot]))
                self.contacts[i_foot].setReference(self.feetGoal[i_foot])
                self.invdyn.addRigidContact(self.contacts[i_foot], self.w_forceRef, 1.0, 1)

        if np.abs(t - 0.2) < 0.0001 or (t >= 0.2 and t < 0.5):
            if np.abs(t - 0.2) < 0.0001:
                self.dz = 0.0003  # Rise
                self.velz = 0.3
            elif np.abs(t - 0.35) < 0.0001:
                self.dz = - 0.0003  # Fall
                self.velz = - 0.3

            for i_foot in [1, 2]:
                self.feetGoal[i_foot].translation = np.matrix(
                    [self.feetGoal[i_foot].translation[0, 0],
                     self.feetGoal[i_foot].translation[1, 0],
                     self.feetGoal[i_foot].translation[2, 0] + self.dz]).T
                self.feetTraj[i_foot] = tsid.TrajectorySE3Constant("traj_FR_foot", self.feetGoal[i_foot])
                self.sampleFoot = self.feetTraj[i_foot].computeNext()
                self.sampleFoot.vel(np.array([[0.0, 0.0, self.velz, 0.0, 0.0, 0.0]]).transpose())
                if np.abs(t - 0.499) < 0.0001:
                    self.sampleFoot.vel(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).transpose())
                self.feetTask[i_foot].setReference(self.sampleFoot)"""

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

        # self.feetTask[1].setReference(self.sampleFoot)

        # pos_foot = self.robot.framePosition(self.data, self.model.getFrameId('FR_FOOT'))
        # print("FOOT: ", pos_foot.translation.T)

        if False:
            pos_base = self.robot.framePosition(self.data, self.model.getFrameId('base_link'))
            self.FR_foot_goal.translation = np.matrix(
                [pos_base.translation[0, 0] + 0.19,
                 pos_base.translation[1, 0] - 0.15005,
                 pos_base.translation[2, 0] - 0.25]).T
            self.trajFRfoot = tsid.TrajectorySE3Constant("traj_FR_foot", self.FR_foot_goal)
            self.sampleFoot = self.trajFRfoot.computeNext()
            self.FRfootTask.setReference(self.sampleFoot)

        # Sinusoidal motion for the foot target
        """if (t > 0):

            FRgoalz = 0.05 * np.sin(self.omega*t) + (self.FR_foot_ref.translation[2, 0] + 0.2)
            pos_base = self.robot.framePosition(self.data, self.model.getFrameId('base_link'))
            print("Pos base:", pos_base)
            self.FR_foot_goal.translation = np.matrix(
                [pos_base.translation[0, 0] + 0.19,
                 pos_base.translation[1, 0] - 0.15005,
                 pos_base.translation[2, 0] - 0.3 + 0.05 * np.sin(self.omega*t)]).T

            self.trajFRfoot = tsid.TrajectorySE3Constant("traj_FR_foot", self.FR_foot_goal)

            self.sampleFoot = self.trajFRfoot.computeNext()
            self.FRfootTask.setReference(self.sampleFoot)"""

        # Resolution of the HQP problem
        HQPData = self.invdyn.computeProblemData(t, self.qdes, self.vdes)
        self.sol = self.solver.solve(HQPData)

        # Torques, accelerations, velocities and configuration computation
        tau_ff = self.invdyn.getActuatorForces(self.sol)
        self.ades = self.invdyn.getAccelerations(self.sol)
        self.vdes += self.ades * dt
        self.qdes = pin.integrate(self.model, self.qdes, self.vdes * dt)

        # Get contact forces for debug purpose
        ctc_forces = self.invdyn.getContactForces(self.sol)
        nb_feet = int(ctc_forces.shape[0] / 3)
        for i_foot in range(nb_feet):
            print("Contact forces foot ", i_foot, ": ", ctc_forces[(i_foot*3):(i_foot*3+3), 0].transpose())
        for i, name in enumerate(['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']):
            pos_foot = self.robot.framePosition(self.data, self.model.getFrameId(name))
            print("Foot ", i, "at position ", pos_foot.translation.transpose())
            print(i, " desired at position ", self.feetGoal[i].translation.transpose())

        # Display non-locked target footholds with green spheres (gepetto gui)
        rgbt = [0.0, 1.0, 0.0, 0.5]
        for i in range(4):
            if (t == 0):
                solo.viewer.gui.addSphere("world/sphere"+str(i)+"_nolock", .02, rgbt)  # .1 is the radius
            solo.viewer.gui.applyConfiguration(
                "world/sphere"+str(i)+"_nolock", (self.feetGoal[i].translation[0, 0],
                                                  self.feetGoal[i].translation[1, 0],
                                                  self.feetGoal[i].translation[2, 0], 1., 0., 0., 0.))

        # Refresh gepetto gui with TSID desired joint position
        solo.display(self.qdes)

        # print("BASE: ", self.qdes[0:3].T)
        """print("Position: ", self.FRfootTask.position[0:3].T)
        print("Target:   ", self.FRfootTask.position_ref[0:3].T)
        print("Error:    ", self.FRfootTask.position_error[0:3].T)"""

        # Torque PD controller
        P = 50  # 50
        D = 0.2
        torques12 = P * (self.qdes[7:] - qmes12[7:]) + D * (self.vdes[6:] - vmes12[6:]) + tau_ff

        # Saturation to limit the maximal torque
        t_max = 2.5
        tau = np.maximum(np.minimum(torques12, t_max * np.ones((12, 1))), -t_max * np.ones((12, 1)))

        # self.error = self.error or (self.sol.status != 0) or (qmes12[8] < -np.pi/2) or (qmes12[11] < -np.pi/2) or (qmes12[14] < -np.pi/2) or (
        #    qmes12[17] < -np.pi/2) or (qmes12[8] > np.pi/2) or (qmes12[11] > np.pi/2) or (qmes12[14] > np.pi/2) or (qmes12[17] > np.pi/2)

        return tau.flatten()

# Parameters for the controller


dt = 0.001				# controller time step

q0 = np.zeros((19, 1))  # initial configuration

omega = 1.0				# sinus pulsation
