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
        mu = 1  				# friction coefficient
        fMin = 1.0				# minimum normal force
        fMax = 100.0  			# maximum normal force
        contactNormal = np.matrix([0., 0., 1.]).T  # direction of the normal to the contact surface

        # Coefficients of the posture task
        kp_posture = 10.0		# proportionnal gain of the posture task
        w_posture = 1.0			# weight of the posture task

        # Coefficients of the contact tasks
        kp_contact = 100.0		    # proportionnal gain for the contacts
        self.w_forceRef = 1e-5		# weight of the forces regularization

        # Coefficients of the foot tracking task
        kp_foot = 1.0		        # proportionnal gain for the tracking task
        self.w_foot = 100000.0		# weight of the tracking task

        # Coefficients of the trunk task
        kp_trunk = np.matrix([0.0, 0.0, 0.0, 50.0, 50.0, 50.0]).T
        w_trunk = 1000

        # Coefficients of the CoM task
        self.kp_com = 300
        self.w_com = 1000
        offset_x_com = - 0.00  # offset along X for the reference position of the CoM

        # Arrays to store logs
        k_max_loop = 4200
        self.f_pos = np.zeros((4, k_max_loop, 3))
        self.f_vel = np.zeros((4, k_max_loop, 3))
        self.f_acc = np.zeros((4, k_max_loop, 3))
        self.f_pos_ref = np.zeros((4, k_max_loop, 3))
        self.f_vel_ref = np.zeros((4, k_max_loop, 3))
        self.f_acc_ref = np.zeros((4, k_max_loop, 3))
        self.b_pos = np.zeros((k_max_loop, 6))

        # Position of the shoulders in local frame
        self.shoulders = np.array([[0.19, 0.19, -0.19, -0.19], [0.15005, -0.15005, 0.15005, -0.15005]])

        # Foot trajectory generator
        max_height_feet = 0.04
        t_lock_before_touchdown = 0.15
        self.ftgs = [ftg.Foot_trajectory_generator(max_height_feet, t_lock_before_touchdown) for i in range(4)]

        # Which pair of feet is active (0 for [1, 2] and 1 for [0, 3])
        self.pair = 0

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

            # Adding the rigid contact after the reference contact force has been set
            self.invdyn.addRigidContact(self.contacts[i], self.w_forceRef)

        #######################
        # FOOT TRACKING TASKS #
        #######################

        self.feetTask = 4*[None]  # List to store the foot tracking tasks
        mask = np.matrix([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).T

        # Task definition (creating the task object)
        for i_foot in range(4):
            self.feetTask[i_foot] = tsid.TaskSE3Equality("foot_track_" + str(i_foot), self.robot, self.foot_frames[i_foot])
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
        mask = np.matrix([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).T
        self.trunkTask.setKp(np.multiply(kp_trunk, mask))
        self.trunkTask.setKd(2.0 * np.sqrt(np.multiply(kp_trunk, mask)))
        self.trunkTask.useLocalFrame(False)

        # Add the task to the HQP with weight = w_trunk, priority level = 1 (not real constraint)
        # and a transition duration = 0.0
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
        x1 = self.shoulders[0, :]
        y1 = self.shoulders[1, :]

        dt = 0.001  #  [s]
        t1 = 0.3  #  [s]

        if pair == 0:
            t0 = (k_loop / 300) * t1
            feet = [1, 2]
        else:
            t0 = ((k_loop-300) / 300) * t1
            feet = [0, 3]

        for i_foot in feet:

            # Get desired 3D position
            [x0, dx0, ddx0,  y0, dy0, ddy0,  z0, dz0, ddz0, gx1, gy1] = (self.ftgs[i_foot]).get_next_foot(
                self.sampleFeet[i_foot].pos()[0, 0], self.sampleFeet[i_foot].vel()[0, 0], self.sampleFeet[i_foot].acc()[1, 0],
                self.sampleFeet[i_foot].pos()[1, 0], self.sampleFeet[i_foot].vel()[1, 0], self.sampleFeet[i_foot].acc()[1, 0],
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

    def control(self, qmes12, vmes12, t, k_simu, solo):

        if k_simu == 0:
            self.qtsid = qmes12
            self.qtsid[:3] = np.zeros((3, 1))  # Discard x and y drift and height position
            self.qtsid[2, 0] = 0.235 - 0.01205385

            self.feetGoal = 4*[None]
            self.sampleFeet = 4*[None]
            self.pos_contact = 4*[None]
            for i_foot in range(4):
                self.feetGoal[i_foot] = self.robot.framePosition(self.invdyn.data(), self.model.getFrameId(self.foot_frames[i_foot]))
                footTraj = tsid.TrajectorySE3Constant("foot_traj", self.feetGoal[i_foot])
                self.sampleFeet[i_foot] = footTraj.computeNext()

                self.pos_contact[i_foot] = np.matrix([self.shoulders[0, i_foot], self.shoulders[1, i_foot], 0.0])
        else:
            # Encoders (position of joints)
            self.qtsid[7:] = qmes12[7:]

            # Gyroscopes (angular velocity of trunk)
            self.vtsid[3:6] = vmes12[3:6]

            # IMU estimation of orientation of the trunk
            self.qtsid[3:7] = qmes12[3:7]

        ################
        # UPDATE TASKS #
        ################

        k_loop = (k_simu - 300) % 600

        if k_simu >= 300:
            if k_loop == 0:  # Start swing phase

                # Update active feet pair
                self.pair = 0

                # Update the foot tracking tasks
                self.update_feet_tasks(k_loop, self.pair)

                for i_foot in [1, 2]:
                    # Disable the contacts for both feet (1 and 2)
                    self.invdyn.removeRigidContact(self.foot_frames[i_foot], 0.0)

                    # Enable the foot tracking task for both feet (1 and 2)
                    self.invdyn.addMotionTask(self.feetTask[i_foot], self.w_foot, 1, 0.0)

                if k_simu >= 900:
                    for i_foot in [0, 3]:
                        # Update the position of the contacts and enable them
                        pos_foot = self.robot.framePosition(
                            self.invdyn.data(), self.model.getFrameId(self.foot_frames[i_foot]))
                        self.pos_contact[i_foot] = pos_foot.translation.transpose()
                        self.contacts[i_foot].setReference(pos_foot)
                        self.invdyn.addRigidContact(self.contacts[i_foot], self.w_forceRef)

                        # Disable both foot tracking tasks
                        self.invdyn.removeTask("foot_track_" + str(i_foot), 0.0)

            elif k_loop > 0 and k_loop < 300:

                # Update the foot tracking tasks
                self.update_feet_tasks(k_loop, self.pair)

            elif k_loop == 300:

                # Update active feet pair
                self.pair = 1

                # Update the foot tracking tasks
                self.update_feet_tasks(k_loop, self.pair)

                for i_foot in [0, 3]:
                    # Disable the contacts for both feet (0 and 3)
                    self.invdyn.removeRigidContact(self.foot_frames[i_foot], 0.0)

                    # Enable the foot tracking task for both feet (0 and 3)
                    self.invdyn.addMotionTask(self.feetTask[i_foot], self.w_foot, 1, 0.0)

                for i_foot in [1, 2]:
                    # Update the position of the contacts and enable them
                    pos_foot = self.robot.framePosition(
                        self.invdyn.data(), self.model.getFrameId(self.foot_frames[i_foot]))
                    self.pos_contact[i_foot] = pos_foot.translation.transpose()
                    self.contacts[i_foot].setReference(pos_foot)
                    self.invdyn.addRigidContact(self.contacts[i_foot], self.w_forceRef)

                    # Disable both foot tracking tasks
                    self.invdyn.removeTask("foot_track_" + str(i_foot), 0.0)
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
            P = 10.0  # 5  # 50
            D = 0.05  # 0.05  #  0.2
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

            # Refresh gepetto gui with TSID desired joint position
            solo.viewer.gui.refresh()
            solo.display(self.qtsid)

        # Log pos, vel, acc of the flying foot
        for i_foot in range(4):
            self.f_pos_ref[i_foot, k_simu:(k_simu+1), :] = self.sampleFeet[i_foot].pos()[0:3].transpose()
            self.f_vel_ref[i_foot, k_simu:(k_simu+1), :] = self.sampleFeet[i_foot].vel()[0:3].transpose()
            self.f_acc_ref[i_foot, k_simu:(k_simu+1), :] = self.sampleFeet[i_foot].acc()[0:3].transpose()

            pos = self.robot.framePosition(self.invdyn.data(), self.model.getFrameId(self.foot_frames[i_foot]))
            vel = self.robot.frameVelocityWorldOriented(self.invdyn.data(), self.model.getFrameId(self.foot_frames[i_foot]))
            acc = self.robot.frameAccelerationWorldOriented(self.invdyn.data(), self.model.getFrameId(self.foot_frames[i_foot]))
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
