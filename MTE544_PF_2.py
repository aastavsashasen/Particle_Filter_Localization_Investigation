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
t0 = timer.time()

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
Q_const_m1 = 0.01  # origionally 0.0133
Q_const_m2 = 0.01
Q = np.array([[Q_const_m1, 0.0, 0.0], [0.0, Q_const_m1, 0.0], [0.0, 0.0, Q_const_m1]])
Q_m2 = np.array([[Q_const_m2, 0.0, 0.0], [0.0, Q_const_m2, 0.0], [0.0, 0.0, Q_const_m2]])

# ------------------------------- Calculating the value of R/Q
rq_m1 = np.trace(R)/np.trace(Q)
rq_m2 = np.trace(R_m2)/np.trace(Q_m2)

# --------------------------------- Create a 2D map
pf_map = np.ones((50, 50)).astype(object)  # the prior belief (50X50)
# within each map grid cell, there is a list of particles
# each particle consists of an array (x, y, theta). The grid is split as follows
dx = 0.1
dy = 0.1
# we limit the number of particles (=1000)
NP = 100
# the rate of particle generation (=1) allow us to keep the map populated
generation_number = 1
# only if a particles weight is above 0.9, then that particle will generate particles
particle_weight_boundary = 0.90
# first place a single particle in every map grid cell. in each cell we have a list of tuples
counter = 0
for row_index, row in enumerate(pf_map):
    for col_index, lis in enumerate(row):
        pf_map[row_index, col_index] = [(col_index*dx, row_index*dy, 0.0, 1.0)]  # x, y, theta, weight
        # pf_map[row_index, col_index].append((col_index*dx, row_index*dy, 45.0*(np.pi/180), 1.0))
        # pf_map[row_index, col_index].append((col_index * dx, row_index * dy, 90.0 * (np.pi / 180), 1.0))
        # pf_map[row_index, col_index].append((col_index * dx, row_index * dy, 135.0 * (np.pi / 180), 1.0))
        # pf_map[row_index, col_index].append((col_index * dx, row_index * dy, 180.0 * (np.pi / 180), 1.0))
        # pf_map[row_index, col_index].append((col_index * dx, row_index * dy, 225.0 * (np.pi / 180), 1.0))
        # pf_map[row_index, col_index].append((col_index * dx, row_index * dy, 270.0 * (np.pi / 180), 1.0))
        # pf_map[row_index, col_index].append((col_index * dx, row_index * dy, 315.0 * (np.pi / 180), 1.0))
        print("updated...")

# not let us plot the grid! (should be an even spread)
# fig = plt.figure()
# for row_index, row in enumerate(pf_map):
#     for col_index, lis in enumerate(row):
#         print("plotted: ", (row_index, col_index))
#         print("state/ lis: ", lis[0])
#         plt.scatter(lis[0][0], lis[0][1], alpha=0.8, edgecolors='none', s=30)
# plt.title('MAP')
# axes = plt.gca()
# axes.set_xticks(np.arange(0, pf_map.shape[0]*dx, dx))
# axes.set_yticks(np.arange(0, pf_map.shape[1]*dy, dy))
# axes.set_xlim([0.0, pf_map.shape[0]*dx])
# axes.set_ylim([0.0, pf_map.shape[1]*dy])
# plt.grid()
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.show()

robot_pos = robot_init_pos
fig1 = plt.figure()
fig1.canvas.draw()
plt.show(block=False)

# store the paths....
robot_path_x = np.array([0.0])
robot_path_y = np.array([0.0])
estimate_path_x = np.array([0.0])
estimate_path_y = np.array([0.0])
estimate_var_x = np.array([0.0])
estimate_var_y = np.array([0.0])

loop_count = 0
while time <= 3:  # original = 3 seconds
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
    robot_path_x = np.append(robot_path_x, robot_pos[0])
    robot_path_y = np.append(robot_path_y, robot_pos[1])
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

    # Now we ensure that we are below the max particle number before generating new particles
    particle_number = 0
    for row_index, row in enumerate(pf_map):
        for col_index, lis in enumerate(row):
            for elem_index, elem in enumerate(lis):
                particle_number = particle_number + 1
    print("The number of particles: ", particle_number)
    while particle_number > NP:  # delete low weight particles
        min_weight = 1.0
        for row_index, row in enumerate(pf_map):
            for col_index, lis in enumerate(row):
                for elem_index, elem in enumerate(lis):
                    if elem[3] < min_weight:
                        min_weight = elem[3]
                        min_particle = np.array([row_index, col_index, elem_index])
        print("element to delete: ", pf_map[min_particle[0]][min_particle[1]][min_particle[2]])
        del pf_map[min_particle[0]][min_particle[1]][min_particle[2]]
        particle_number = particle_number - 1

    pf_map_fixed = pf_map.copy()  # for generating particles and avoiding generating from generated particles
    for row_index, row in enumerate(pf_map_fixed):
        for col_index, lis in enumerate(row):
            for elem_index, elem in enumerate(lis):
                if elem[3] > particle_weight_boundary:  # should do this for all particles with the prob to generate
                                                        # being proportional to the weight
                    print("Particle x,y: ", (elem[0], elem[1]))
                    # now generate particle around this particle...
                    for i in range(0, generation_number):  # no need to generate within range either...
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
                            # give the generated particle the same weight as the particle it came from
                            pf_map[y_index_temp, x_index_temp].append((x_temp, y_temp, theta_temp, elem[3]))
                            particle_number = particle_number + 1
                        else:
                            print("generated particle OUT OF RANGE")
    print("Final number of particles: ", particle_number)
    # now calculate a state estimate
    # do this by modelling a gaussian for the robot pose
    # ie) get a mean and variance
    x_sum = 0.0
    x_divider = 0.0
    y_sum = 0.0
    y_divider = 0.0
    x_vals = np.array([0.0])
    y_vals = np.array([0.0])
    weights_array = np.array([0.0])
    estimate_array = np.array([0.0, 0.0, 0.0, 0.0])
    particle_count = 0
    for row_index, row in enumerate(pf_map):
        for col_index, lis in enumerate(row):
            for elem_index, elem in enumerate(lis):
                # here the average/ mean (=estimated pose) is weighted
                x_sum = x_sum + elem[0]*elem[3]
                x_divider = x_divider + elem[3]
                y_sum = y_sum + elem[1]*elem[3]
                y_divider = y_divider + elem[3]
                x_vals = np.append(x_vals, elem[0])
                weights_array = np.append(weights_array, elem[3])
                y_vals = np.append(y_vals, elem[1])
                particle_count = particle_count + 1
    #x_mean = x_sum/particle_count
    x_mean = x_sum/x_divider
    #y_mean = y_sum/particle_count
    y_mean = y_sum/y_divider
    #x_var = np.var(x_vals[1:])
    x_var = np.sqrt(np.cov(x_vals[1:], aweights=weights_array[1:]))
    #y_var = np.var(y_vals[1:])
    y_var = np.sqrt(np.cov(y_vals[1:], aweights=weights_array[1:]))
    # we attempt to estimate the robot position only!
    state_estimate = np.array([x_mean, y_mean, x_var, y_var])
    #print("Thus the x/y mean and x/y variance (respectively): ", state_estimate)
    # save the estimated path.....
    estimate_path_x = np.append(estimate_path_x, state_estimate[0])
    estimate_path_y = np.append(estimate_path_y, state_estimate[1])
    estimate_var_x = np.append(estimate_var_x, state_estimate[2])
    estimate_var_y = np.append(estimate_var_y, state_estimate[3])

    error_sum = 0.0
    # Calculate the error in the position estimate
    error_sum = error_sum + abs(robot_pos[0] - state_estimate[0]) + abs(robot_pos[1] - state_estimate[1])
    # now we can PLOT the new particle distribution map!
    print(">>>>>>>>>>>>>>>> completed update loop with time: ", round(time, 1))
    # fig1 = plt.figure()
    if (loop_count % 3) == 0 or loop_count == 0 or loop_count == 1:
        for row_index, row in enumerate(pf_map):
            for col_index, lis in enumerate(row):
                for elem_index, elem in enumerate(lis):
                    plt.scatter(elem[0], elem[1], alpha=1.0*elem[3], color='red', edgecolors='none', s=10*(elem[3]*15))
        plt.scatter(robot_pos[0], robot_pos[1], alpha=1.0, color='blue', edgecolors='none', s=90, label='Robot Pose')
        plt.scatter(state_estimate[0], state_estimate[1], alpha=1.0, color='green', edgecolors='none', s=90, label='Pose Estimate')
        map_string = "MAP || time-step = {}".format(round(time, 1))
        plt.title(map_string)
        plt.legend()
        axes = plt.gca()
        axes.set_xticks(np.arange(0, pf_map.shape[0]*dx, dx))
        axes.set_yticks(np.arange(0, pf_map.shape[1]*dy, dy))
        axes.set_xlim([0.0, pf_map.shape[0]*dx])
        axes.set_ylim([0.0, pf_map.shape[1]*dy])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid()
        fig1.canvas.draw()
        print(">>>>>>>>>>>>>>>> CUMULATIVE ESTIMATION ERROR = ", error_sum)
        timer.sleep(0.1)
        plt.clf()
        #plt.show()


    # increment time step...
    time = round(time + time_step, 1)
    time_array = np.append(time_array, np.array([time]))
    # increment the loop count (helps plot every 3 steps)
    loop_count = loop_count + 1
    print("==================================== TIME-STEP: ", time)
# ===================================================================
t1 = timer.time()
print("Total Time taken: ", (t1-t0))
print(">>>>>> TOTAL ESTIMATION ERROR: ", error_sum)
print(">>>>>>>>>>>>>>>>> SIMULATION COMPLETE")
print("Averaged x variance: ", np.mean(estimate_var_x[1:]))
print("Averaged y variance: ", np.mean(estimate_var_y[1:]))

print("...... now we can plot the entire paths...")
fig_final = plt.figure()
ax = plt.axes()
ax.plot(robot_path_x[1:], robot_path_y[1:], label='True Path', linestyle='-', marker='x')
ax.plot(estimate_path_x[1:], estimate_path_y[1:], label='Estimated Path', linestyle='--', marker='o')
# plotting y variance
y = estimate_path_y[1:]
error = np.sqrt(estimate_var_y[1:])
plt.fill_between(estimate_path_x[1:], y-error, y+error, alpha=0.1, color='red', label='Estimate Single Standard Deviation')
# plotting x variance
x = estimate_path_x[1:]
error2 = np.sqrt(estimate_var_x[1:])
plt.fill_betweenx(estimate_path_y[1:], x-error2, x+error2, alpha=0.1, color='red')
final_string = "Final Path(s): Particle Number = {}".format(NP)
plt.title(final_string)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

print("=================== CODE COMPLETE ==================")
