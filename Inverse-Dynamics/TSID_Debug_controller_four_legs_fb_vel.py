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
import foot_trajectory_generator as ftg
import FootstepPlanner
import pybullet as pyb

pin.switchToNumpyMatrix()


########################################################################
#            Class for a PD with feed-forward Controller               #
########################################################################

class controller:

    def __init__(self, q0, omega, t):

        self.q_ref = np.array([[0.0, 0.0, 0.235 - 0.01205385, 0.0, 0.0, 0.0, 1.0,
                                0.0, 0.8, -1.6, 0, 0.8, -1.6,
                                0, -0.8, 1.6, 0, -0.8, 1.6]]).transpose()

        self.qtsid = self.q_ref.copy()
        self.vtsid = np.zeros((18, 1))
        self.ades = np.zeros((18, 1))

        self.error = False
        self.verbose = True

        # List with the names of all feet frames
        self.foot_frames = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']

        # Constraining the contacts
        mu = 2  				# friction coefficient
        fMin = 1.0				# minimum normal force
        fMax = 100.0  			# maximum normal force
        contactNormal = np.matrix([0., 0., 1.]).T  # direction of the normal to the contact surface

        # Coefficients of the posture task
        kp_posture = 10.0		# proportionnal gain of the posture task
        w_posture = 1.0         # weight of the posture task

        # Coefficients of the contact tasks
        kp_contact = 100.0         # proportionnal gain for the contacts
        self.w_forceRef = 10000.0  # weight of the forces regularization

        # Coefficients of the foot tracking task
        kp_foot = 1.0               # proportionnal gain for the tracking task
        self.w_foot = 10000.0       # weight of the tracking task

        # Coefficients of the trunk task
        kp_trunk = np.matrix([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).T
        w_trunk = 30.0

        # Coefficients of the CoM task
        self.kp_com = 300
        self.w_com = 1000.0  #  1000.0
        offset_x_com = - 0.00  # offset along X for the reference position of the CoM

        # Arrays to store logs
        k_max_loop = 72000
        self.f_pos = np.zeros((4, k_max_loop, 3))
        self.f_vel = np.zeros((4, k_max_loop, 3))
        self.f_acc = np.zeros((4, k_max_loop, 3))
        self.f_pos_ref = np.zeros((4, k_max_loop, 3))
        self.f_vel_ref = np.zeros((4, k_max_loop, 3))
        self.f_acc_ref = np.zeros((4, k_max_loop, 3))
        self.b_pos = np.zeros((k_max_loop, 6))

        # Position of the shoulders in local frame
        self.shoulders = np.array([[0.19, 0.19, -0.19, -0.19], [0.15005, -0.15005, 0.15005, -0.15005]])
        self.footsteps = self.shoulders.copy()
        self.memory_contacts = self.shoulders.copy()

        # Foot trajectory generator
        max_height_feet = 0.04
        t_lock_before_touchdown = 0.15
        self.ftgs = [ftg.Foot_trajectory_generator(max_height_feet, t_lock_before_touchdown) for i in range(4)]

        # Which pair of feet is active (0 for [1, 2] and 1 for [0, 3])
        self.pair = -1

        # Rotation along the vertical axis
        delta_yaw = (2 * np.pi / 10) * t
        c, s = np.cos(delta_yaw), np.sin(delta_yaw)
        self.R_yaw = np.array([[c, s], [-s, c]])

        # Footstep planner object
        self.fstep_planner = FootstepPlanner.FootstepPlanner(0.03, self.shoulders, 0.001)
        self.v_ref = np.zeros((6, 1))
        self.vu_m = np.zeros((6, 1))
        self.t_stance = 0.3
        self.T_gait = 0.6
        self.t_remaining = np.zeros((1, 4))
        self.h_ref = 0.235 - 0.01205385

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
        self.invdyn.computeProblemData(t, self.qtsid, self.vtsid)

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
        self.trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", self.q_ref[7:])
        self.samplePosture = self.trajPosture.computeNext()
        self.postureTask.setReference(self.samplePosture)

        ############
        # CONTACTS #
        ############

        self.contacts = 4*[None]  # List to store the rigid contact tasks

        for i, name in enumerate(self.foot_frames):

            # Contact definition (creating the contact object)
            self.contacts[i] = tsid.ContactPoint(name, self.robot, name, contactNormal, mu, fMin, fMax)
            self.contacts[i].setKp((kp_contact * matlib.ones(3).T))
            self.contacts[i].setKd((2.0 * np.sqrt(kp_contact) * matlib.ones(3).T))
            self.contacts[i].useLocalFrame(False)

            # Set the contact reference position
            H_ref = self.robot.framePosition(self.invdyn.data(), self.model.getFrameId(name))
            H_ref.translation = np.matrix(
                [H_ref.translation[0, 0],
                 H_ref.translation[1, 0],
                 0.0]).T
            self.contacts[i].setReference(H_ref)

            w_reg_f = 1
            if i in [0, 1]:
                self.contacts[i].setForceReference(np.matrix([0.0, 0.0, w_reg_f * 14.0]).T)
            else:
                self.contacts[i].setForceReference(np.matrix([0.0, 0.0, w_reg_f * 17.0]).T)
            self.contacts[i].setRegularizationTaskWeightVector(np.matrix([w_reg_f, w_reg_f, w_reg_f]).T)

            # Adding the rigid contact after the reference contact force has been set
            self.invdyn.addRigidContact(self.contacts[i], self.w_forceRef)

        #######################
        # FOOT TRACKING TASKS #
        #######################

        self.feetTask = 4*[None]  # List to store the foot tracking tasks
        mask = np.matrix([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).T

        # Task definition (creating the task object)
        for i_foot in range(4):
            self.feetTask[i_foot] = tsid.TaskSE3Equality(
                "foot_track_" + str(i_foot), self.robot, self.foot_frames[i_foot])
            self.feetTask[i_foot].setKp(kp_foot * mask)
            self.feetTask[i_foot].setKd(2.0 * np.sqrt(kp_foot) * mask)
            self.feetTask[i_foot].setMask(mask)
            self.feetTask[i_foot].useLocalFrame(False)

        # The reference will be set later when the task is enabled

        ######################
        # TRUNK POSTURE TASK #
        ######################

        # Task definition (creating the task object)
        self.trunkTask = tsid.TaskSE3Equality("task-trunk", self.robot, 'base_link')
        mask = np.matrix([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).T
        self.trunkTask.setKp(np.multiply(kp_trunk, mask))
        self.trunkTask.setKd(2.0 * np.sqrt(np.multiply(kp_trunk, mask)))
        self.trunkTask.useLocalFrame(False)
        self.trunkTask.setMask(mask)

        # Add the task to the HQP with weight = w_trunk, priority level = 1 (not real constraint)
        # and a transition duration = 0.0
        # if w_trunk > 0.0:
        self.invdyn.addMotionTask(self.trunkTask, w_trunk, 1, 0.0)

        # TSID Trajectory (creating the trajectory object and linking it to the task)
        self.trunk_ref = self.robot.framePosition(self.invdyn.data(), self.model.getFrameId('base_link'))
        self.trajTrunk = tsid.TrajectorySE3Constant("traj_base_link", self.trunk_ref)
        self.sampleTrunk = self.trajTrunk.computeNext()
        self.sampleTrunk.pos(np.matrix([0.0, 0.0, 0.235 - 0.01205385, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).T)
        self.sampleTrunk.vel(np.matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T)
        self.sampleTrunk.acc(np.matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T)
        self.trunkTask.setReference(self.sampleTrunk)

        ############
        # COM TASK #
        ############

        # Task definition
        self.comTask = tsid.TaskComEquality("task-com", self.robot)
        self.comTask.setKp(self.kp_com * matlib.ones(3).T)
        self.comTask.setKd(2.0 * np.sqrt(self.kp_com) * matlib.ones(3).T)
        if self.w_com > 0.0:
            self.invdyn.addMotionTask(self.comTask, self.w_com, 1, 0.0)

        # Task reference
        com_ref = self.robot.com(self.invdyn.data())
        self.trajCom = tsid.TrajectoryEuclidianConstant("traj_com", com_ref)
        self.sample_com = self.trajCom.computeNext()

        tmp = self.sample_com.pos()  # Temp variable to store CoM position
        tmp[0, 0] += offset_x_com
        self.sample_com.pos(tmp)
        self.comTask.setReference(self.sample_com)

        ##########
        # SOLVER #
        ##########

        # Use EiquadprogFast solver
        self.solver = tsid.SolverHQuadProgFast("qp solver")

        # Resize the solver to fit the number of variables, equality and inequality constraints
        self.solver.resize(self.invdyn.nVar, self.invdyn.nEq, self.invdyn.nIn)

    ####################################################################
    #           Method to updated desired foot position                #
    ####################################################################

    def update_feet_tasks(self, k_loop, pair):

        # Target (x, y) positions for both feet
        x1 = self.footsteps[0, :]
        y1 = self.footsteps[1, :]

        dt = 0.001  #  [s]
        t1 = 0.28  #  [s]

        if pair == -1:
            return 0
        elif pair == 0:
            t0 = ((k_loop-20) / 280) * t1
            feet = [1, 2]
        else:
            t0 = ((k_loop-320) / 280) * t1
            feet = [0, 3]

        for i_foot in feet:

            # Get desired 3D position
            [x0, dx0, ddx0,  y0, dy0, ddy0,  z0, dz0, ddz0, gx1, gy1] = (self.ftgs[i_foot]).get_next_foot(
                self.sampleFeet[i_foot].pos()[0, 0], self.sampleFeet[i_foot].vel()[
                    0, 0], self.sampleFeet[i_foot].acc()[1, 0],
                self.sampleFeet[i_foot].pos()[1, 0], self.sampleFeet[i_foot].vel()[
                    1, 0], self.sampleFeet[i_foot].acc()[1, 0],
                x1[i_foot], y1[i_foot], t0,  t1, dt)

            # Get sample object
            footTraj = tsid.TrajectorySE3Constant("foot_traj", self.feetGoal[i_foot])
            self.sampleFeet[i_foot] = footTraj.computeNext()

            # Update desired pos, vel, acc
            self.sampleFeet[i_foot].pos(np.matrix([x0, y0, z0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).T)
            self.sampleFeet[i_foot].vel(np.matrix([dx0, dy0, dz0, 0.0, 0.0, 0.0]).T)
            self.sampleFeet[i_foot].acc(np.matrix([ddx0, ddy0, ddz0, 0.0, 0.0, 0.0]).T)

            # Set reference
            self.feetTask[i_foot].setReference(self.sampleFeet[i_foot])

            # Update footgoal for display purpose
            self.feetGoal[i_foot].translation = np.matrix([x0, y0, z0]).T

        return 0

    ####################################################################
    #                      Torque Control method                       #
    ####################################################################

    def control(self, qmes12, vmes12, t, k_simu, solo, mpc):

        if k_simu == 0:
            self.qtsid = qmes12
            self.qtsid[:3] = np.zeros((3, 1))  # Discard x and y drift and height position
            self.qtsid[2, 0] = 0.235 - 0.01205385

            self.feetGoal = 4*[None]
            self.sampleFeet = 4*[None]
            self.pos_contact = 4*[None]
            for i_foot in range(4):
                self.feetGoal[i_foot] = self.robot.framePosition(
                    self.invdyn.data(), self.model.getFrameId(self.foot_frames[i_foot]))
                footTraj = tsid.TrajectorySE3Constant("foot_traj", self.feetGoal[i_foot])
                self.sampleFeet[i_foot] = footTraj.computeNext()

                self.pos_contact[i_foot] = np.matrix([self.footsteps[0, i_foot], self.footsteps[1, i_foot], 0.0])
        """else:
            # Encoders (position of joints)
            self.qtsid[7:] = qmes12[7:]

            # Gyroscopes (angular velocity of trunk)
            self.vtsid[3:6] = vmes12[3:6]

            # IMU estimation of orientation of the trunk
            self.qtsid[3:7] = qmes12[3:7]"""

        #####################
        # FOOTSTEPS PLANNER #
        #####################

        k_loop = (k_simu - 0) % 600

        for i_foot in [1, 2]:
            self.t_remaining[0, i_foot] = np.max((0.0, 0.3 * (300 - k_loop) * 0.001))
        for i_foot in [0, 3]:
            if k_loop < 300:
                self.t_remaining[0, i_foot] = 0.0
            else:
                self.t_remaining[0, i_foot] = 0.3 * (600 - k_loop) * 0.001

        # Get PyBullet velocity in local frame
        """RPY = pyb.getEulerFromQuaternion(qmes12[3:7])
        c, s = np.cos(RPY[2]), np.sin(RPY[2])
        R = np.array([[c, s], [-s, c]])
        self.vu_m[0:2, 0:1] = np.dot(R, vmes12[0:2,0:1])"""

        """if k_simu == 1000:
            self.vu_m[0:2, 0:1] = np.array([[0.0, 0.1]]).transpose()
            self.vtsid[0:2, 0:1] = np.array([[0.0, 0.1]]).transpose()
        elif k_simu > 1000:
            self.vu_m[0:2, 0:1] = self.vtsid[0:2, 0:1].copy()"""

        if k_simu == 1500:
            self.vu_m[0:2, 0:1] = np.array([[0.1, 0.0]]).transpose()
            self.v_ref[0:2, 0:1] = np.array([[0.1, 0.0]]).transpose()
        """if k_simu == 6000:
            self.vu_m[0:2, 0:1] = np.array([[0.0, 0.0]]).transpose()
            self.v_ref[0:2, 0:1] = np.array([[0.0, 0.0]]).transpose()"""

        """RPY = pyb.getEulerFromQuaternion(self.qtsid[3:7])
        c, s = np.cos(-RPY[2]), np.sin(-RPY[2])
        R = np.array([[c, s], [-s, c]])
        self.vtsid[0:2, 0:1] = np.dot(R, self.vu_m[0:2, 0:1])"""

        # Update desired location of footsteps using the footsteps planner
        self.fstep_planner.update_footsteps(self.v_ref, self.vu_m, self.t_stance,
                                            self.t_remaining, self.T_gait, self.h_ref)

        self.footsteps = self.memory_contacts + self.fstep_planner.footsteps

        # Rotate footsteps depending on TSID orientation
        """RPY = pyb.getEulerFromQuaternion(self.qtsid[3:7])
        c, s = np.cos(RPY[2]), np.sin(RPY[2])
        R = np.array([[c, s], [-s, c]])
        self.footsteps = np.dot(R, self.footsteps)"""

        #############################
        # UPDATE ROTATION ON ITSELF #
        #############################

        # self.footsteps = np.dot(self.R_yaw, self.footsteps)

        #######################
        # UPDATE CoM POSITION #
        #######################

        tmp = self.sample_com.pos()  # Temp variable to store CoM position
        tmp[0, 0] = np.mean(self.footsteps[0, :])
        tmp[1, 0] = np.mean(self.footsteps[1, :])
        self.sample_com.pos(tmp)
        """if k_simu >= 1500 and k_simu < 2000:
            tmp = self.sample_com.vel()
            tmp[0, 0] = + 0.1 * np.min((1.0, 1.0 - (2000 - k_simu) / 500))
            self.sample_com.vel(tmp)"""
        self.comTask.setReference(self.sample_com)

        """self.sampleTrunk.pos(np.matrix([tmp[0, 0], tmp[1, 0], 0.235 - 0.01205385,
                                        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).T)
        self.trunkTask.setReference(self.sampleTrunk)"""

        # print("Desired position of CoM: ", tmp.transpose())

        #####################################
        # UPDATE REFERENC OF CONTACT FORCES #
        #####################################

        # TODO: Remove "w_reg_f *" in setForceReference once the tsid bug is fixed

        """w_reg_f = 1000.0
        if k_loop >= 320:
            for j, i_foot in enumerate([1, 2]):
                self.contacts[i_foot].setForceReference(w_reg_f * np.matrix(mpc.f_applied[3*j:3*(j+1)]).T)
                self.contacts[i_foot].setRegularizationTaskWeightVector(np.matrix([w_reg_f, w_reg_f, w_reg_f]).T)
        elif k_loop >= 300:
            for j, i_foot in enumerate([0, 1, 2, 3]):
                self.contacts[i_foot].setForceReference(w_reg_f * np.matrix(mpc.f_applied[3*j:3*(j+1)]).T)
                self.contacts[i_foot].setRegularizationTaskWeightVector(np.matrix([w_reg_f, w_reg_f, w_reg_f]).T)
        elif k_loop >= 20:
            for j, i_foot in enumerate([0, 3]):
                self.contacts[i_foot].setForceReference(w_reg_f * np.matrix(mpc.f_applied[3*j:3*(j+1)]).T)
                self.contacts[i_foot].setRegularizationTaskWeightVector(np.matrix([w_reg_f, w_reg_f, w_reg_f]).T)
        else:
            for j, i_foot in enumerate([0, 1, 2, 3]):
                self.contacts[i_foot].setForceReference(w_reg_f * np.matrix(mpc.f_applied[3*j:3*(j+1)]).T)
                self.contacts[i_foot].setRegularizationTaskWeightVector(np.matrix([w_reg_f, w_reg_f, w_reg_f]).T)"""

        ################
        # UPDATE TASKS #
        ################

        # To follow a sinus in pitch then roll
        """if k_simu >= 6000:
            c, s = 0.2 * np.cos((k_simu - 300) * 0.001 * 2 * np.pi * 0.2 + np.pi * 0.5), \
                0.2 * np.sin((k_simu - 300) * 0.001 * 2 * np.pi * 0.2 + np.pi * 0.5)
            self.sampleTrunk.pos(np.matrix([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c]).T)
            self.trunkTask.setReference(self.sampleTrunk)
        elif k_simu >= 300:
            c, s = 0.2 * np.cos((k_simu - 300) * 0.001 * 2 * np.pi * 0.2 + np.pi * 0.5), \
                0.2 * np.sin((k_simu - 300) * 0.001 * 2 * np.pi * 0.2 + np.pi * 0.5)
            self.sampleTrunk.pos(np.matrix([0.0, 0.0, 0.0, c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c]).T)
            self.trunkTask.setReference(self.sampleTrunk)"""

        # To follow a sinus both in pitch and roll
        """if k_simu >= 300:
            c, s = 0.2 * np.cos((k_simu - 300) * 0.001 * 2 * np.pi * 0.1 + np.pi * 0.5), \
                0.2 * np.sin((k_simu - 300) * 0.001 * 2 * np.pi * 0.1 + np.pi * 0.5)
            R1 = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
            c, s = 0.2 * np.cos((k_simu - 300) * 0.001 * 2 * np.pi * 0.1 + np.pi * 0.5), \
                0.2 * np.sin((k_simu - 300) * 0.001 * 2 * np.pi * 0.1 + np.pi * 0.5)
            R2 = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
            c, s = 0.2 * np.cos((k_simu - 300) * 0.001 * 2 * np.pi * 0.4 + np.pi * 0.5), \
                0.2 * np.sin((k_simu - 300) * 0.001 * 2 * np.pi * 0.4 + np.pi * 0.5)
            R3 = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            R = np.dot(R2, R1)
            self.sampleTrunk.pos(np.matrix([0.0, 0.0, 0.0, R[0, 0], R[0, 1], R[0, 2],
                                            R[1, 0], R[1, 1], R[1, 2], R[2, 0], R[2, 1], R[2, 2]]).T,)
            self.trunkTask.setReference(self.sampleTrunk)"""

        if k_simu >= 0:
            if k_loop == 0:  # Start swing phase

                # Update active feet pair
                self.pair = -1

                # Update the foot tracking tasks
                self.update_feet_tasks(k_loop, self.pair)

                if k_simu >= 600:
                    for i_foot in [0, 3]:
                        # Update the position of the contacts and enable them
                        pos_foot = self.robot.framePosition(
                            self.invdyn.data(), self.model.getFrameId(self.foot_frames[i_foot]))
                        self.pos_contact[i_foot] = pos_foot.translation.transpose()
                        self.memory_contacts[:, i_foot] = self.footsteps[:,
                                                                         i_foot]  #  pos_foot.translation[0:2].flatten()
                        self.contacts[i_foot].setReference(pos_foot)
                        self.invdyn.addRigidContact(self.contacts[i_foot], self.w_forceRef)

                        # Disable both foot tracking tasks
                        self.invdyn.removeTask("foot_track_" + str(i_foot), 0.0)

            elif k_loop == 20:

                # Update active feet pair
                self.pair = 0

                # Update the foot tracking tasks
                self.update_feet_tasks(k_loop, self.pair)

                for i_foot in [1, 2]:
                    # Disable the contacts for both feet (1 and 2)
                    self.invdyn.removeRigidContact(self.foot_frames[i_foot], 0.0)

                    # Enable the foot tracking task for both feet (1 and 2)
                    self.invdyn.addMotionTask(self.feetTask[i_foot], self.w_foot, 1, 0.0)

            elif k_loop > 20 and k_loop < 300:

                # Update the foot tracking tasks
                self.update_feet_tasks(k_loop, self.pair)

            elif k_loop == 300:

                # Update active feet pair
                self.pair = -1

                # Update the foot tracking tasks
                self.update_feet_tasks(k_loop, self.pair)

                for i_foot in [1, 2]:
                    # Update the position of the contacts and enable them
                    pos_foot = self.robot.framePosition(
                        self.invdyn.data(), self.model.getFrameId(self.foot_frames[i_foot]))
                    self.pos_contact[i_foot] = pos_foot.translation.transpose()
                    self.memory_contacts[:, i_foot] = self.footsteps[:, i_foot]  #  pos_foot.translation[0:2].flatten()
                    self.contacts[i_foot].setReference(pos_foot)
                    self.invdyn.addRigidContact(self.contacts[i_foot], self.w_forceRef)

                    # Disable both foot tracking tasks
                    self.invdyn.removeTask("foot_track_" + str(i_foot), 0.0)

            elif k_loop == 320:

                # Update active feet pair
                self.pair = 1

                # Update the foot tracking tasks
                self.update_feet_tasks(k_loop, self.pair)

                for i_foot in [0, 3]:
                    # Disable the contacts for both feet (0 and 3)
                    self.invdyn.removeRigidContact(self.foot_frames[i_foot], 0.0)

                    # Enable the foot tracking task for both feet (0 and 3)
                    self.invdyn.addMotionTask(self.feetTask[i_foot], self.w_foot, 1, 0.0)

            else:

                # Update the foot tracking tasks
                self.update_feet_tasks(k_loop, self.pair)

        ###############
        # HQP PROBLEM #
        ###############

        # Resolution of the HQP problem
        HQPData = self.invdyn.computeProblemData(t, self.qtsid, self.vtsid)
        self.sol = self.solver.solve(HQPData)

        # Torques, accelerations, velocities and configuration computation
        tau_ff = self.invdyn.getActuatorForces(self.sol)
        self.fc = self.invdyn.getContactForces(self.sol)
        print(k_simu, " : ", self.fc.transpose())
        # print(self.fc.transpose())
        self.ades = self.invdyn.getAccelerations(self.sol)
        self.vtsid += self.ades * dt
        self.qtsid = pin.integrate(self.model, self.qtsid, self.vtsid * dt)

        # Call display and log function
        self.display_and_log(t, solo, k_simu)

        # Placeholder torques for PyBullet
        tau = np.zeros((12, 1))

        # Check for NaN value
        if np.any(np.isnan(tau_ff)):
            # self.error = True
            tau = np.zeros((12, 1))
        else:
            # Torque PD controller
            P = 3.0  # 10.0  # 5  # 50
            D = 0.3  # 0.05  # 0.05  #  0.2
            torques12 = P * (self.qtsid[7:] - qmes12[7:]) + D * (self.vtsid[6:] - vmes12[6:]) + tau_ff

            # Saturation to limit the maximal torque
            t_max = 2.5
            tau = np.clip(torques12, -t_max, t_max)  # faster than np.maximum(a_min, np.minimum(a, a_max))

        return tau.flatten()

    def display_and_log(self, t, solo, k_simu):

        if self.verbose:
            # Display target 3D positions of footholds with green spheres (gepetto gui)
            rgbt = [0.0, 1.0, 0.0, 0.5]
            for i in range(0, 4):
                if (t == 0):
                    solo.viewer.gui.addSphere("world/sphere"+str(i)+"_target", .02, rgbt)  # .1 is the radius
                solo.viewer.gui.applyConfiguration(
                    "world/sphere"+str(i)+"_target", (self.feetGoal[i].translation[0, 0],
                                                      self.feetGoal[i].translation[1, 0],
                                                      self.feetGoal[i].translation[2, 0], 1., 0., 0., 0.))

            # Display current 3D positions of footholds with magenta spheres (gepetto gui)
            rgbt = [1.0, 0.0, 1.0, 0.5]
            for i in range(0, 4):
                if (t == 0):
                    solo.viewer.gui.addSphere("world/sphere"+str(i)+"_pos", .02, rgbt)  # .1 is the radius
                pos_foot = self.robot.framePosition(self.invdyn.data(), self.model.getFrameId(self.foot_frames[i]))
                solo.viewer.gui.applyConfiguration(
                    "world/sphere"+str(i)+"_pos", (pos_foot.translation[0, 0],
                                                   pos_foot.translation[1, 0],
                                                   pos_foot.translation[2, 0], 1., 0., 0., 0.))

            # Display target 3D positions of footholds with green spheres (gepetto gui)
            rgbt = [0.0, 0.0, 1.0, 0.5]
            for i in range(0, 4):
                if (t == 0):
                    solo.viewer.gui.addSphere("world/shoulder"+str(i), .02, rgbt)  # .1 is the radius
                solo.viewer.gui.applyConfiguration(
                    "world/shoulder"+str(i), (self.shoulders[0, i], self.shoulders[1, i], 0.0, 1., 0., 0., 0.))

            # Display 3D positions of sampleFeet
            """rgbt = [0.3, 1.0, 1.0, 0.5]
            for i in range(0, 4):
                if (t == 0):
                    solo.viewer.gui.addSphere("world/sfeet"+str(i), .02, rgbt)  # .1 is the radius
                solo.viewer.gui.applyConfiguration(
                    "world/sfeet"+str(i), (self.sampleFeet[i].pos()[0, 0],
                                           self.sampleFeet[i].pos()[1, 0],
                                           self.sampleFeet[i].pos()[2, 0], 1., 0., 0., 0.))"""

            # Display lines for contact forces
            """if (t == 0):
                for i in range(4):
                    solo.viewer.gui.addCurve("world/force_curve"+str(i),
                                             [[0., 0., 0.], [0., 0., 0.]], [1.0, 0.0, 0.0, 0.5])
                    solo.viewer.gui.setCurveLineWidth("world/force_curve"+str(i), 8.0)
                    solo.viewer.gui.setColor("world/force_curve"+str(i), [1.0, 0.0, 0.0, 0.5])
            else:
                if self.pair == 1:
                    feet = [1, 2]
                    feet_0 = [0, 3]
                else:
                    feet = [0, 3]
                    feet_0 = [1, 2]

                for i, i_foot in enumerate(feet):
                    Kreduce = 0.04
                    solo.viewer.gui.setCurvePoints("world/force_curve"+str(i_foot),
                                                   [[self.memory_contacts[0, i_foot],
                                                     self.memory_contacts[1, i_foot], 0.0],
                                                    [self.memory_contacts[0, i_foot] + Kreduce * self.fc[3*i+0, 0],
                                                     self.memory_contacts[1, i_foot] + Kreduce * self.fc[3*i+1, 0],
                                                     Kreduce * self.fc[3*i+2, 0]]])
                for i, i_foot in enumerate(feet_0):
                    solo.viewer.gui.setCurvePoints("world/force_curve"+str(i_foot),
                                                   [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])"""

            """if (t == 0):
                solo.viewer.gui.addCurve("world/orientation_curve",
                                         [[0., 0., 0.], [0., 0., 0.]], [1.0, 0.0, 0.0, 0.5])
                solo.viewer.gui.setCurveLineWidth("world/orientation_curve", 8.0)
                solo.viewer.gui.setColor("world/orientation_curve", [1.0, 0.0, 0.0, 0.5])

            pos_trunk = self.robot.framePosition(self.invdyn.data(), self.model.getFrameId("base_link"))
            line_rot = np.dot(pos_trunk.rotation, np.array([[1, 0, 0]]).transpose())
            solo.viewer.gui.setCurvePoints("world/orientation_curve",
                                           [pos_trunk.translation.flatten().tolist()[0],
                                            (pos_trunk.translation + line_rot).flatten().tolist()[0]])"""

            """if k_simu == 0:
                solo.viewer.gui.setRefreshIsSynchronous(False)"""

            # Refresh gepetto gui with TSID desired joint position
            if k_simu % 1 == 0:
                solo.viewer.gui.refresh()
                solo.display(self.qtsid)

        # Log pos, vel, acc of the flying foot
        for i_foot in range(4):
            self.f_pos_ref[i_foot, k_simu:(k_simu+1), :] = self.sampleFeet[i_foot].pos()[0:3].transpose()
            self.f_vel_ref[i_foot, k_simu:(k_simu+1), :] = self.sampleFeet[i_foot].vel()[0:3].transpose()
            self.f_acc_ref[i_foot, k_simu:(k_simu+1), :] = self.sampleFeet[i_foot].acc()[0:3].transpose()

            pos = self.robot.framePosition(self.invdyn.data(), self.model.getFrameId(self.foot_frames[i_foot]))
            vel = self.robot.frameVelocityWorldOriented(
                self.invdyn.data(), self.model.getFrameId(self.foot_frames[i_foot]))
            acc = self.robot.frameAccelerationWorldOriented(
                self.invdyn.data(), self.model.getFrameId(self.foot_frames[i_foot]))
            self.f_pos[i_foot, k_simu:(k_simu+1), :] = pos.translation[0:3].transpose()
            self.f_vel[i_foot, k_simu:(k_simu+1), :] = vel.vector[0:3].transpose()
            self.f_acc[i_foot, k_simu:(k_simu+1), :] = acc.vector[0:3].transpose()

        # Log position of the base
        pos_trunk = self.robot.framePosition(self.invdyn.data(), self.model.getFrameId("base_link"))
        self.b_pos[k_simu:(k_simu+1), 0:3] = pos_trunk.translation[0:3].transpose()

# Parameters for the controller


dt = 0.001				# controller time step

q0 = np.zeros((19, 1))  # initial configuration

omega = 1  # Not used
