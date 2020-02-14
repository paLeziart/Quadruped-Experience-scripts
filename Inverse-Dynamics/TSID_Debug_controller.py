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

        self.q_ref = np.array([[0.0, 0.0, 0.235 - 0.01264513, 0.0, 0.0, 0.0, 1.0,
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
        kp_foot = 100.0		        # proportionnal gain for the tracking task
        self.w_foot = 1000.0		# weight of the tracking task

        # Coefficients of the trunk task
        kp_trunk = np.matrix([20000, 20000, 20000, 10000, 10000, 10000]).T
        w_trunk = 1000

        # Coefficients of the CoM task
        self.kp_com = 100
        self.w_com = 1000
        offset_x_com = - 0.05  # offset along X for the reference position of the CoM

        # Arrays to store logs
        k_max_loop = 1200
        self.f_pos = np.zeros((k_max_loop, 3))
        self.f_vel = np.zeros((k_max_loop, 3))
        self.f_acc = np.zeros((k_max_loop, 3))
        self.f_pos_ref = np.zeros((k_max_loop, 3))
        self.f_vel_ref = np.zeros((k_max_loop, 3))
        self.f_acc_ref = np.zeros((k_max_loop, 3))
        self.b_pos = np.zeros((k_max_loop, 6))

        # Position of the shoulders in local frame
        self.shoulders = np.array([[0.19, 0.19, -0.19, -0.19], [0.15005, -0.15005, 0.15005, -0.15005]])

        # Foot trajectory generator
        max_height_feet = 0.04
        t_lock_before_touchdown = 0.15
        self.ftg = ftg.Foot_trajectory_generator(max_height_feet, t_lock_before_touchdown)

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

        ######################
        # FOOT TRACKING TASK #
        ######################

        # Task definition (creating the task object)
        self.footTask = tsid.TaskSE3Equality("foot_track", self.robot, "FR_FOOT")
        mask = np.matrix([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).T
        self.footTask.setKp(kp_foot * mask)
        self.footTask.setKd(2.0 * np.sqrt(kp_foot) * mask)
        self.footTask.setMask(mask)
        self.footTask.useLocalFrame(False)

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

        # Add the task to the HQP with weight = w_trunk, priority level = 1 (not real constraint)
        # and a transition duration = 0.0
        #self.invdyn.addMotionTask(self.trunkTask, w_trunk, 1, 0.0)

        # TSID Trajectory (creating the trajectory object and linking it to the task)
        self.trunk_ref = self.robot.framePosition(self.invdyn.data(), self.model.getFrameId('base_link'))
        self.trajTrunk = tsid.TrajectorySE3Constant("traj_base_link", self.trunk_ref)
        self.sampleTrunk = self.trajTrunk.computeNext()
        self.sampleTrunk.pos(np.matrix([0.0, 0.0, 0.235 - 0.01264513, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).T)
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

    def update_foot_task(self, k_loop):

        # Target (x, y) position
        x1 = 0.19
        y1 = -0.15005

        dt = 0.001  #  [s]
        t1 = 0.3  #  [s]
        t0 = (k_loop / 300) * t1

        # Get desired 3D position
        [x0, dx0, ddx0,  y0, dy0, ddy0,  z0, dz0, ddz0, gx1, gy1] = self.ftg.get_next_foot(
            self.sampleFoot.pos()[0, 0], self.sampleFoot.vel()[0, 0], self.sampleFoot.acc()[1, 0],
            self.sampleFoot.pos()[1, 0], self.sampleFoot.vel()[1, 0], self.sampleFoot.acc()[1, 0],
            x1, y1, t0,  t1, dt)

        # Get sample object
        self.footTraj = tsid.TrajectorySE3Constant("foot_traj", self.footGoal)
        self.sampleFoot = self.footTraj.computeNext()

        # Update desired pos, vel, acc
        self.sampleFoot.pos(np.matrix([x0, y0, z0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).T)
        self.sampleFoot.vel(np.matrix([dx0, dy0, dz0, 0.0, 0.0, 0.0]).T)
        self.sampleFoot.acc(np.matrix([ddx0, ddy0, ddz0, 0.0, 0.0, 0.0]).T)

        # Set reference
        self.footTask.setReference(self.sampleFoot)

        # Update footgoal for display purpose
        self.footGoal.translation = np.matrix([x0, y0, z0]).T

        return 0

    ####################################################################
    #                      Torque Control method                       #
    ####################################################################

    def control(self, qmes12, vmes12, t, k_simu, solo):

        if k_simu == 0:
            self.qtsid = qmes12
            self.qtsid[:3] = np.zeros((3, 1))  # Discard x and y drift and height position
            self.qtsid[2, 0] = 0.235 - 0.01264513

            self.footGoal = self.robot.framePosition(self.invdyn.data(), self.model.getFrameId("FR_FOOT"))
            self.footTraj = tsid.TrajectorySE3Constant("foot_traj", self.footGoal)
            self.sampleFoot = self.footTraj.computeNext()

            self.pos_contact = np.matrix([0.19, -0.15005, 0.0])

        ################
        # UPDATE TASKS #
        ################

        k_loop = k_simu % 600

        if k_loop == 0:  # Start swing phase

            # Disable the contact
            self.invdyn.removeRigidContact("FR_FOOT", 0.0)

            # Update the foot tracking task
            self.update_foot_task(k_loop)

            # Enable the foot tracking task
            self.invdyn.addMotionTask(self.footTask, self.w_foot, 1, 0.0)

        elif k_loop < 300:

            # Update the foot tracking task
            self.update_foot_task(k_loop)

        elif k_loop == 300:

            # Update the position of the contact and enable it
            pos_foot = self.robot.framePosition(
                self.invdyn.data(), self.model.getFrameId(self.foot_frames[1]))
            self.pos_contact = pos_foot.translation.transpose()
            self.contacts[1].setReference(pos_foot)
            self.invdyn.addRigidContact(self.contacts[1], self.w_forceRef)

            # Disable the foot tracking task
            self.invdyn.removeTask("foot_track", 0.0)

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

        return tau.flatten()

    def display_and_log(self, t, solo, k_simu):

        if self.verbose:
            # Display target 3D positions of footholds with green spheres (gepetto gui)
            rgbt = [0.0, 1.0, 0.0, 0.5]
            for i in range(1, 2):
                if (t == 0):
                    solo.viewer.gui.addSphere("world/sphere"+str(i)+"_target", .02, rgbt)  # .1 is the radius
                solo.viewer.gui.applyConfiguration(
                    "world/sphere"+str(i)+"_target", (self.footGoal.translation[0, 0],
                                                      self.footGoal.translation[1, 0],
                                                      self.footGoal.translation[2, 0], 1., 0., 0., 0.))

            # Display current 3D positions of footholds with magenta spheres (gepetto gui)
            rgbt = [1.0, 0.0, 1.0, 0.5]
            for i in range(1, 2):
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
        self.f_pos_ref[k_simu:(k_simu+1), :] = self.sampleFoot.pos()[0:3].transpose()
        self.f_vel_ref[k_simu:(k_simu+1), :] = self.sampleFoot.vel()[0:3].transpose()
        self.f_acc_ref[k_simu:(k_simu+1), :] = self.sampleFoot.acc()[0:3].transpose()

        pos = self.robot.framePosition(self.invdyn.data(), self.model.getFrameId("FR_FOOT"))
        vel = self.robot.frameVelocityWorldOriented(self.invdyn.data(), self.model.getFrameId("FR_FOOT"))
        acc = self.robot.frameAccelerationWorldOriented(self.invdyn.data(), self.model.getFrameId("FR_FOOT"))
        self.f_pos[k_simu:(k_simu+1), :] = pos.translation[0:3].transpose()
        self.f_vel[k_simu:(k_simu+1), :] = vel.vector[0:3].transpose()
        self.f_acc[k_simu:(k_simu+1), :] = acc.vector[0:3].transpose()

        pos_trunk = self.robot.framePosition(self.invdyn.data(), self.model.getFrameId("base_link"))
        self.b_pos[k_simu:(k_simu+1), 0:3] = pos_trunk.translation[0:3].transpose()

# Parameters for the controller


dt = 0.001				# controller time step

q0 = np.zeros((19, 1))  # initial configuration

omega = 1  # Not used
