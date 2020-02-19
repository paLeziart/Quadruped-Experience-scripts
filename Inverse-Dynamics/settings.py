# coding: utf8

import numpy as np

# def init():
#     global n_contacts

n_contacts = np.array([])
dt = 0.02
v_ref = np.array([[0.1, 0.00, 0.0, 0, 0, 0.0]]).T
qu_m = np.array([[0, 0, 0.25, 0, 0, 0]]).T
vu_m = np.array([[0.0, 0.00, 0.0, 0, 0, 0.4]]).T
t_stance = 0.3
T_gait = 0.6
# FL, FR, HL, HR
phases = np.array([[0, 0.5, 0.5, 0]])
x_ref = np.array([])
p_contacts = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
t = 0.0
shoulders = np.array([[0.5, 0.5, -0.5, -0.5], [0.5, -0.5, 0.5, -0.5]])
footholds = np.array([[0.5, 0.5, -0.5, -0.5], [0.5, -0.5, 0.5, -0.5]])

shoulders = np.array(
    [[0.19, 0.19, -0.19, -0.19], [0.1046, -0.1046, 0.1046, -0.1046]])
footholds = np.array(
    [[0.19, 0.19, -0.19, -0.19], [0.1046, -0.1046, 0.1046, -0.1046]])
