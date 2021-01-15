#import autograd.numpy as np
import numpy as np
import time as timer
import autograd
from autograd import jacobian
import matplotlib.pyplot as plt
import matplotlib
import _tkinter
import scipy
from scipy.stats import norm
matplotlib.rcParams['figure.figsize'] = (15.0, 10.0)

# on using autograd to find Jacobian:
# https://stackoverflow.com/questions/49553006/compute-the-jacobian-matrix-in-python
# =============================================================

# The inputs are custom
def u1_const (t):
    return 2.0

def u2_const (t):
    return 2.0

def u1_rand_sin (t):
    return np.random.normal(0, 0.1, 1)[0] + np.sin(t)

def u2_rand_sin (t):
    return np.random.normal(0, 0.1, 1)[0] + np.cos(t)

# The system is as follows
def x1_calc (x, t, input_func):
    return x[0] + input_func(t)*np.cos(x[2])*time_step

def x2_calc (x, t, input_func):
    return x[1] + input_func(t)*np.sin(x[2])*time_step

def x3_calc (x, t, input_func2):
    return x[2] + input_func2(t)*time_step

# ================== first measurement model....
def y1_calc (x):
    return x[0]

def y2_calc (x):
    return x[1]

def y3_calc (x):
    return x[2]

def y1_calc_noise (x):
    return x[0] + np.random.normal(0, R[0, 0], 1)[0]

def y2_calc_noise (x):
    return x[1] + np.random.normal(0, R[1, 1], 1)[0]

def y3_calc_noise (x):
    return x[2] + np.random.normal(0, R[2, 2], 1)[0]

# ================== second measurement model....
def y1_calc_m2 (x):
    return np.sqrt(np.square(x[0])+np.square(x[1]))

def y2_calc_m2 (x):
    return np.tan(x[2])

def y1_calc_noise_m2 (x):
    return np.sqrt(np.square(x[0])+np.square(x[1])) + np.random.normal(0, R_m2[0, 0], 1)[0]

def y2_calc_noise_m2 (x):
    return np.tan(x[2]) + np.random.normal(0, R_m2[1, 1], 1)[0]

# The conditions are
x_0 = 3.0
y_0 = 2.0
q_0 = 45.0*(np.pi/180)
robot_init_pos = np.array([x_0, y_0, q_0])

baseline = 1.24
max_time = 20.0
time_step = 0.1

time = 0.0
time_array = np.array([0.0])
x1 = x_0
x2 = y_0
x3 = q_0
x_state = np.array([x1, x2, x3], dtype=float)
x_state_m2 = np.array([x1, x2, x3], dtype=float)
y1 = 0.0
y2 = 0.0
x_state_cov = np.zeros((3, 3))
x_state_cov_m2 = np.zeros((3, 3))
# save arrays for later plotting
predicted_states = np.array([0, 0, 0])
corrected_states = np.array([0, 0, 0])
predicted_states_m2 = np.array([0, 0, 0])
corrected_states_m2 = np.array([0, 0, 0])
# for uncertainty/ noise modelling
# R for measurement => GIVEN!
R = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]])
R_m2 = np.array([[0.5, 0.0], [0.0, 1.0]])
# Q for model. 0.5, 0.5, 0.1
Q_const_m1 = 0.0133
Q_const_m2 = 0.01
Q = np.array([[Q_const_m1, 0.0, 0.0], [0.0, Q_const_m1, 0.0], [0.0, 0.0, Q_const_m1]])
Q_m2 = np.array([[Q_const_m2, 0.0, 0.0], [0.0, Q_const_m2, 0.0], [0.0, 0.0, Q_const_m2]])

# ------------------------------- Calculating the value of R/Q
rq_m1 = np.trace(R)/np.trace(Q)
rq_m2 = np.trace(R_m2)/np.trace(Q_m2)

# --------------------------------- Create a 2D map
pf_map = np.ones((50, 50)).astype(object)  # the prior belief
# within each map grid cell, there is a list of particles
# each particle consists of an array (x, y, theta). The grid is split as follows
dx = 0.1
dy = 0.1
# the rate of particle generation allow us to keep the map populated
generation_number = 1
# only if a particles weight is above 0.8, then that particle will generate particles
particle_weight_boundary = 0.8
# first place a single particle in every map grid cell. in each cell we have a list of tuples
counter = 0
for row_index, row in enumerate(pf_map):
    for col_index, lis in enumerate(row):
        pf_map[row_index, col_index] = [(col_index*dx, row_index*dy, 0.0, 1.0)]  # x, y, theta, weight
        print("updated...")

# not let us plot the grid! (should be an even spread)
#fig = plt.figure()
#for row_index, row in enumerate(pf_map):
#    for col_index, lis in enumerate(row):
#        print("plotted: ", (row_index, col_index))
#        print("state/ lis: ", lis[0])
#        plt.scatter(lis[0][0], lis[0][1], alpha=0.8, edgecolors='none', s=30)
#plt.title('MAP')
#plt.legend()
#plt.show()

robot_pos = robot_init_pos
fig1 = plt.figure()
fig1.canvas.draw()
plt.show(block=False)

while time <= 20:
    # ======================================================= Particle Filter

    # make all the weights of every particle = 1
    for row_index, row in enumerate(pf_map):
        for col_index, lis in enumerate(row):
            for elem_index, elem in enumerate(lis):
                lis[elem_index] = (elem[0], elem[1], elem[2], 1.0)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> PREDICTION
    # actual position of the robot?
    robot_temp_actual_pos = np.array([x1_calc(robot_pos, time, u1_const),
                                 x2_calc(robot_pos, time, u1_const),
                                 x3_calc(robot_pos, time, u2_const)])
    robot_pos = robot_temp_actual_pos
    # first step all particles forward using the motion model
    for row_index, row in enumerate(pf_map):
        for col_index, lis in enumerate(row):
            #if len(lis) == 0:
            #    break
            for elem_index, elem in enumerate(lis):
                print("list element as tuple: ", elem)
                print("old point: ", elem)
                particle_forward = np.array([[x1_calc(elem, time, u1_const) + np.random.normal(0, Q[0, 0], 1)[0]],
                                         [x2_calc(elem, time, u1_const) + np.random.normal(0, Q[1, 1], 1)[0]],
                                         [x3_calc(elem, time, u2_const) + np.random.normal(0, Q[2, 2], 1)[0]]])
                print("delete element: ", lis[elem_index])
                del lis[elem_index]
                # now place the new position of the particle on the map
                x_temp = round(particle_forward[0][0], 1)
                x_index_temp = int(round(x_temp/dx, 0))
                #print("x index: ", x_index_temp)
                #print("x value: ", x_temp)
                y_temp = round(particle_forward[1][0], 1)
                y_index_temp = int(round(y_temp/dy, 0))
                theta_temp = particle_forward[2][0]
                #print("map shape: ", pf_map.shape)
                print("new point: ", (x_temp, y_temp, theta_temp, 1.0))
                # Now calculate the weight for the point
                state_temp_y = np.array([x_temp, y_temp, theta_temp])
                y_state_pred_noise = np.array([y1_calc_noise(robot_pos),
                                               y2_calc_noise(robot_pos),
                                               y3_calc_noise(robot_pos)])
                # now calculate the weight.....
                weights = norm.pdf(y_state_pred_noise, state_temp_y, R)
                weight = np.trace(weights)
                print("weight assigned: ", weight)
                # if we are still on the map, save point
                if y_index_temp < pf_map.shape[0] and x_index_temp < pf_map.shape[1]:
                    pf_map[y_index_temp, x_index_temp].append((x_temp, y_temp, theta_temp, weight))
                    #pf_map[y_index_temp, x_index_temp] = [(x_temp, y_temp, theta_temp, 1.0)]
    # Now to go through the entire map and normalize the weights
    # first summing all the weights....
    weight_max = 0.0
    for row_index, row in enumerate(pf_map):
        for col_index, lis in enumerate(row):
            for elem_index, elem in enumerate(lis):
                if elem[3] > weight_max:
                    weight_max = elem[3]
    print(">>>>> MAX OF WEIGHTS: ", weight_max)
    # then dividing all the weights by the max found weight
    for row_index, row in enumerate(pf_map):
        for col_index, lis in enumerate(row):
            for elem_index, elem in enumerate(lis):
                temp_weight = elem[3] / weight_max
                lis[elem_index] = (elem[0], elem[1], elem[2], temp_weight)
                print("final Normalized (by weight) element: ", lis[elem_index])
    # Now we sample from this distribution to get new particles
    # if its probable we keep the particle, otherwise remove it
    for row_index, row in enumerate(pf_map):
        for col_index, lis in enumerate(row):
            for elem_index, elem in enumerate(lis):
                if elem[3] > np.random.normal(0, 1, 1):
                    print("keep particle, weight = ", elem[3])
                else:
                    del lis[elem_index]

    # now generate particles from existing particles!
    pf_map_fixed = pf_map.copy()  # for generating particles and avoiding generating from generated particles
    for row_index, row in enumerate(pf_map_fixed):
        for col_index, lis in enumerate(row):
            for elem_index, elem in enumerate(lis):
                if elem[3] > particle_weight_boundary:
                    print("Particle x,y: ", (elem[0], elem[1]))
                    # now generate particle around this particle...
                    for i in range(0, generation_number):
                        del_pos_x = np.random.normal(0, 0.1, 1)[0]
                        del_pos_y = np.random.normal(0, 0.1, 1)[0]

                        x_temp = round(elem[0] + del_pos_x, 1)
                        print("x_temp: ", x_temp)
                        x_index_temp = int(round(x_temp / dx, 0))
                        print("x_index_temp: ", x_index_temp)
                        y_temp = round(elem[1] + del_pos_y, 1)
                        print("y_temp: ", y_temp)
                        y_index_temp = int(round(y_temp / dy, 0))
                        print("y_index_temp: ", y_index_temp)
                        theta_temp = elem[2]
                        if 0 <= y_index_temp < pf_map.shape[0] and 0 <= x_index_temp < pf_map.shape[1]:
                            pf_map[y_index_temp, x_index_temp].append((x_temp, y_temp, theta_temp, 1.0))
                        else:
                            print("generated particle OUT OF RANGE")

    # now we can PLOT the new particle distribution map!
    print(">>>>>>>>>>>>>>>> completed first update loop")
    # fig1 = plt.figure()
    for row_index, row in enumerate(pf_map):
        for col_index, lis in enumerate(row):
            for elem_index, elem in enumerate(lis):
                #print("elem: ", elem)
                #print("plotted: ", (row_index, col_index))
                #print("state/ lis: ", lis)
                plt.scatter(elem[0], elem[1], alpha=1.0*elem[3], color='red', edgecolors='none', s=10*(elem[3]*15), label='Particles')
    plt.scatter(robot_pos[0], robot_pos[1], alpha=1.0, color='blue', edgecolors='none', s=90, label='Robot Pose')
    plt.title('MAP')
    plt.legend()
    axes = plt.gca()
    axes.set_xticks(np.arange(0, pf_map.shape[0]*dx, dx))
    axes.set_yticks(np.arange(0, pf_map.shape[1]*dy, dy))
    axes.set_xlim([0.0, pf_map.shape[0]*dx])
    axes.set_ylim([0.0, pf_map.shape[1]*dy])
    plt.grid()
    fig1.canvas.draw()

    timer.sleep(1)
    plt.clf()
    #plt.draw()
    #plt.show()
    #time.sleep(0.1)
    #plt.close('all')

    # increment time step...
    time = time + time_step
    time_array = np.append(time_array, np.array([time]))
    print("==================================== TIME-STEP: ", time)


print(">>>>>>>>>>>>>>>>> SIMULATION COMPLETE, now plotting....")



