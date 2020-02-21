# coding: utf8

import numpy as np

# def init():
#     global n_contacts

n_contacts = np.array([])
dt = 0.005
v_ref = np.array([[0.1, 0.00, 0.0, 0, 0, 0.0]]).T
qu_m = np.array([[0, 0, 0.2027, 0, 0, 0]]).T  # 0.235 - 0.01205385
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
    [[0.19, 0.19, -0.19, -0.19], [0.15005, -0.15005, 0.15005, -0.15005]])
footholds = np.array(
    [[0.19, 0.19, -0.19, -0.19], [0.15005, -0.15005, 0.15005, -0.15005]])
