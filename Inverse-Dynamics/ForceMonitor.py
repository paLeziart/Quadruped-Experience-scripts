# coding: utf8

import numpy as np


class ForceMonitor:

    def __init__(self, p, robotId, planeId):

        self.pyb_simu = p
        self.lines = []
        self.robotId = robotId
        self.planeId = planeId

    def getContactPoint(self, contactPoints):
        """ Sort contacts points as there should be only one contact per foot
            and sometimes PyBullet detect several of them. There is one contact
            with a non zero force and the others have a zero contact force
        """

        for i in range(0, len(contactPoints)):
            # There may be several contact points for each foot but only one of them as a non zero normal force
            if (contactPoints[i][9] != 0):
                return contactPoints[i]

        # If it returns 0 then it means there is no contact point with a non zero normal force (should not happen)
        return 0

    def display_contact_forces(self):

        # Info about contact points with the ground
        contactPoints_FL = self.pyb_simu.getContactPoints(self.robotId, self.planeId, linkIndexA=3)  # Front left  foot
        contactPoints_FR = self.pyb_simu.getContactPoints(self.robotId, self.planeId, linkIndexA=7)  # Front right foot
        contactPoints_HL = self.pyb_simu.getContactPoints(self.robotId, self.planeId, linkIndexA=11)  # Hind left  foot
        contactPoints_HR = self.pyb_simu.getContactPoints(self.robotId, self.planeId, linkIndexA=15)  # Hind right foot
        # print(len(contactPoint_FL), len(contactPoint_FR), len(contactPoint_HL), len(contactPoint_HR))

        # Sort contacts points to get only one contact per foot
        contactPoints = []
        contactPoints.append(self.getContactPoint(contactPoints_FL))
        contactPoints.append(self.getContactPoint(contactPoints_FR))
        contactPoints.append(self.getContactPoint(contactPoints_HL))
        contactPoints.append(self.getContactPoint(contactPoints_HR))

        # Display debug lines for contact forces visualization
        i_line = 0
        # print(len(self.lines))
        f_tmp = [0.0] * 3
        for contact in contactPoints:
            if not isinstance(contact, int):  # type(contact) != type(0):
                start = [contact[6][0], contact[6][1], contact[6][2]]
                end = [contact[6][0], contact[6][1], contact[6][2]]
                K = 0.02
                for i_direction in range(0, 3):
                    f_tmp[i_direction] = (contact[9] * contact[7][i_direction] + contact[10] *
                                          contact[11][i_direction] + contact[12] * contact[13][i_direction])
                    end[i_direction] += K * f_tmp[i_direction]

                if contact[3] < 10:
                    print("Link  ", contact[3], "| Contact force: (", f_tmp[0], ", ", f_tmp[1], ", ", f_tmp[2], ")")
                else:
                    print("Link ", contact[3], "| Contact force: (", f_tmp[0], ", ", f_tmp[1], ", ", f_tmp[2], ")")

                if (i_line+1) > len(self.lines):  # If not enough existing lines in line storage a new item is created
                    lineID = self.pyb_simu.addUserDebugLine(start, end, lineColorRGB=[1.0, 0.0, 0.0], lineWidth=8)
                    self.lines.append(lineID)
                else:  # If there is already an existing line item we modify it (to avoid flickering)
                    self.lines[i_line] = self.pyb_simu.addUserDebugLine(start, end, lineColorRGB=[
                        1.0, 0.0, 0.0], lineWidth=8, replaceItemUniqueId=self.lines[i_line])
                i_line += 1

        for i_zero in range(i_line, len(self.lines)):
            self.lines[i_zero] = self.pyb_simu.addUserDebugLine([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], lineColorRGB=[
                1.0, 0.0, 0.0], lineWidth=8, replaceItemUniqueId=self.lines[i_zero])
