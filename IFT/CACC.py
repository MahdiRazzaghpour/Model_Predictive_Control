"""
Cooperative Adaptive Cruise Control  Simulator.
"""
# ==============================================================================
# --     Imports      ----------------------------------------------------------
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import cvxpy as cp
from cvxpy import *
import random
import math
import scipy
import scipy.stats as ss
import seaborn as sns
from scipy.io import loadmat
import pickle
# ==============================================================================
# -- Initialization   ----------------------------------------------------------
# ==============================================================================
Number_of_vehicels = 15
inter_vehicle_time = 0.8 #s
communication_error_rate = 0.0 # percent

T = 10    # prediction horizon

CT = 0.1    # communication time tick [s]
DT = 0.1  # simulator time tick [s]
dl = 1.0  # course tick [m]
NX = 4    # x = [x, y, v, yaw]
NU = 2    # a = [accel, steer]

MAX_STEER = np.deg2rad(45.0)    # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)   # maximum steering speed [rad/s]
MAX_SPEED = 120 / 3.6            # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6         # minimum speed [m/s]
MAX_ACCEL = 1                 # maximum accel [m/ss]

N_IND_SEARCH = 10               # Search index number
# ==============================================================================
# --  MPC parameters  ----------------------------------------------------------
# ==============================================================================
R = np.diag([0.01, 0.01])  # input cost matrix  a = [accel, steer]
Rd = np.diag([0.01, 10.0])  # input difference cost matrix  a = [accel, steer]
Q = np.diag([1, 1, 100000, 3]) # state cost matrix  x = [x, y, v, yaw]
Qf = Q  # state final matrix
GOAL_DIS = 30 # goal distance

gap_weight = 200
speed_weight = 300
# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================
def calc_error(vehicle,neighbor):
    delta_d =  np.sqrt((vehicle.x-neighbor.x)**2 + (vehicle.y - neighbor.y)**2)
    value = delta_d/vehicle.v - inter_vehicle_time
    return value
# ==============================================================================
# -- Trajectory Generation -----------------------------------------------------
# ==============================================================================
raw_data = loadmat('trajectory_plot.mat')
cx = raw_data['X']
# cx=cx[0:100,:]
cy = raw_data['Y']
# cy=cy[0:100,:]
sp = raw_data['V']
# sp=sp[0:100,:]
cyaw= raw_data['Yaw']
# cyaw=cyaw[0:100,:]
# ==============================================================================
# -- Classes -------------------------------------------------------------------
# ==============================================================================
class Agent():
    """
    Agent implements a basic vehicle that follows a trajectory and capable of communication.
    """

    def __init__(self, id, x, y, goal):
        """
        Vehicle parameters
        """
        self.ID = id
        self.x = x
        self.hx = self.x
        self.y = y
        self.hy = self.y
        self.v = 15
        self.hv = self.v
        self.a = 0
        self.ha = self.a
        self.yaw = 0
        self.hyaw = self.yaw
        self.ht = 0
        self.hd = 0
        self.goal = goal
        self.steer = 0.0

        self.xref = None
        self.dref = None
        self.target_ind = None
        self.odelta = None
        self.oa = None
        self.leaders = []
        self.error = 0
        self.error_bias=0
# ==============================================================================
# -- Vehicle's physical specifications   ---------------------------------------
# ==============================================================================
        self.LENGTH = 4.5       # [m]
        self.WIDTH = 2.0        # [m]
        self.BACKTOWHEEL = 1.0  # [m]
        self.WHEEL_LEN = 0.3    # [m]
        self.WHEEL_WIDTH = 0.2  # [m]
        self.TREAD = 0.7        # [m]
        self.WB = 2.5           # [m]

    def update_state(self, a, delta):
        """
        updating x, y, v & yaw based on acceleration and steering angle
        a: acceleration, delta: steering angle
        """
        # input check
        if delta >= MAX_STEER:
            delta = MAX_STEER
        elif delta <= -MAX_STEER:
            delta = -MAX_STEER

        self.x = self.x + self.v * np.cos(self.yaw) * DT
        self.y = 0#self.y + self.v * np.sin(self.yaw) * DT
        self.yaw = 0#self.yaw + self.v / self.WB * np.tan(delta) * DT
        self.v = self.v + a * DT

        if self.v > MAX_SPEED:
            self.v = MAX_SPEED
        elif self.v < MIN_SPEED:
            self.v = MIN_SPEED

        return self

    def receive(self, message_list):
        self.leaders = message_list

class MPC:
    """
    MPC control class, solving optimization problem over cvxpy optimizer
    with respect to cost functions and constraints
    """
    def __init__(self):
        self.horizon = T    # prediction horizon
        self.dt = DT

    def predict_motion(vehicle, oa, od, xref):
        """
        predicting states over the horizon
        oa: optimal acc, od: optimal steering angle, xref: reference state
        """
        xbar = xref * 0.0
        xbar[0, 0] = vehicle.x
        xbar[1, 0] = vehicle.y
        xbar[2, 0] = vehicle.v
        xbar[3, 0] = vehicle.yaw

        #x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, T + 1)):
            vehicle.update_state(ai, di)
            xbar[0, i] = vehicle.x
            xbar[1, i] = vehicle.y
            xbar[2, i] = vehicle.v
            xbar[3, i] = vehicle.yaw

        vehicle.x = xbar[0, 0]
        vehicle.y = xbar[1, 0]
        vehicle.v = xbar[2, 0]
        vehicle.yaw = xbar[3, 0]

        return xbar

    def linear_mpc_control(xref, xbar, vehicle, dref, vehicles_list):
        """
        linear mpc control
        xref: reference point
        xbar: operational point
        x0: initial state
        dref: reference steer angle
        """
        x0 = [vehicle.x, vehicle.y, vehicle.v, vehicle.yaw]  # current state
        x = cvxpy.Variable((NX, T + 1))
        u = cvxpy.Variable((NU, T))
        z = cvxpy.Variable()

        cost = 0.0
        constraints = []

        for t in range(T):
            cost += cvxpy.quad_form(u[:, t], R) # input cost function

            for leader in vehicle.leaders:
                if leader.ID < vehicle.ID-1:
                    z = cvxpy.norm2(x[0,:] -leader.x) + cvxpy.norm2(x[1,:]- leader.y)
                    cost += gap_weight**2 * (z - (inter_vehicle_time*(vehicle.ID-leader.ID)*vehicle.v))
                    cost += speed_weight**2 * cvxpy.norm2(x[2,:] - leader.v)
                    #constraints += [z== inter_vehicle_distance*(vehicle.ID-leader.ID)]
                if (vehicle.ID - leader.ID) == 1 :
                    for front in vehicles_list:
                        if front.ID == leader.ID:
                            z = cvxpy.norm2(x[0,:] -front.x) + cvxpy.norm2(x[1,:]- front.y)
                            cost += gap_weight**2 * (z - (inter_vehicle_time*(vehicle.ID-leader.ID)*vehicle.v))
                            cost += speed_weight**2 * cvxpy.norm2(x[2,:] - front.v)
                            #constraints += [z== inter_vehicle_distance*(vehicle.ID-leader.ID)]


            if t != 0:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)    #current state and reference difference

            A, B, C = get_linear_model_matrix(
                xbar[2, t], xbar[3, t], dref[0, t],vehicle)
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

            if t < (T - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)  #input difference cost function
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <=
                                MAX_DSTEER * DT]

        cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)
        #constraints += [z >= inter_vehicle_distance/2*(vehicle.ID-leader.ID) , z <= inter_vehicle_distance*3/2*(vehicle.ID-leader.ID)]
        constraints += [x[:, 0] == x0]
        constraints += [x[2, :] <= MAX_SPEED]
        constraints += [x[2, :] >= MIN_SPEED]
        constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
        constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.GUROBI, verbose=False)#ECOS

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox = get_nparray_from_matrix(x.value[0, :])
            oy = get_nparray_from_matrix(x.value[1, :])
            ov = get_nparray_from_matrix(x.value[2, :])
            oyaw = get_nparray_from_matrix(x.value[3, :])
            oa = get_nparray_from_matrix(u.value[0, :])
            odelta = get_nparray_from_matrix(u.value[1, :])
            #print(get_nparray_from_matrix(z.value))
        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov

class communication():
        def __init__(self, vehicle):
            """
            Vehicle parameters
            """
            self.ID = vehicle.ID
            self.x = vehicle.x
            self.y = vehicle.y
            self.v = vehicle.v
            self.a = vehicle.a
            self.yaw = vehicle.yaw

        def send (self, vehicle):
            if (random.random() <= 1-communication_error_rate):
                self.x = vehicle.x
                self.y = vehicle.y
                self.v = vehicle.v
                self.a = vehicle.a
                self.yaw = vehicle.yaw
            else:
                self.x = self.x + self.v * np.cos(self.yaw) * DT
                self.y = 0#self.y + self.v * np.sin(self.yaw) * DT

# ==============================================================================
# -- Simulation Loop -----------------------------------------------------------
# ==============================================================================
def main():
    time = 0
    counter = 0
    vehicles_list = []
    message_list = []
    error_list = []
    error_list_s = []
    done = False
    initial = np.zeros(Number_of_vehicels)


    for i in range(Number_of_vehicels):
        print(i,(Number_of_vehicels-i-1)*inter_vehicle_time*15)
        vehicle = Agent(i, (Number_of_vehicels-i-1)*inter_vehicle_time*15 , 0, [cx[-1],cy[-1]] )
        vehicles_list.append(vehicle)

        message = communication(vehicle)
        message_list.append(message)

    while done == False:
        plt.cla()
        print(counter)
        for vehicle in vehicles_list:
            if check_goal(vehicle, vehicle.target_ind, len(cx)):
                done = True

            vehicle.calc_ref_trajectory(cx, cy, cyaw, sp, dl)

            if (counter % 1) == 0 :
                vehicle.receive(message_list)

            oa, odelta, ox, oy, oyaw, ov = MPC.iterative_linear_mpc_control(vehicle.xref, vehicle, vehicle.dref, vehicle.oa, vehicle.odelta,vehicles_list)

            if odelta is not None:
                di, ai = odelta[0], oa[0]

            vehicle.update_state(ai, di)
            vehicle.hx = np.append(vehicle.x, vehicle.hx)
            vehicle.hy = np.append(vehicle.y, vehicle.hy)
            vehicle.hyaw = np.append(vehicle.yaw, vehicle.hyaw)
            vehicle.hv = np.append(vehicle.v, vehicle.hv)
            vehicle.ht = np.append(time, vehicle.ht)
            vehicle.hd = np.append(di, vehicle.hd)
            vehicle.ha = np.append(ai, vehicle.ha)

            for leader in vehicles_list:
                if (vehicle.ID - leader.ID) == 1 :
                    error = calc_error(vehicle,leader)
                    # if counter == 0:
                    #     vehicle.error_bias=error
                    vehicle.error = np.append(error , vehicle.error)
                    error_list.append(np.abs(error*vehicle.v))
                    error_list_s.append(np.abs(error))


            if (counter % 1) == 0.0:
                for message in message_list:
                    if message.ID == vehicle.ID:
                        message.send(vehicle)

        time = time + DT
        counter = counter + 1

    plt.close("all")
    plt.subplots()
    plt.plot(cx, cy, "-r", label="Trajectory")
    for vehicle in vehicles_list:
        plt.plot(vehicle.hx, vehicle.hy, label="vehicle[%d]'s trajectory"%vehicle.ID)
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()
    plt.savefig('Trajectory.eps', format='eps')
    plt.savefig('Trajectory.png', format='png')

    plt.subplots()
    for vehicle in vehicles_list:
        if vehicle.ID != 0 :
            plt.plot(vehicle.ht, vehicle.error, label="vehicle[%d]'s error"%vehicle.ID)
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("error [s]")
    plt.legend()
    plt.savefig('Error.eps', format='eps')
    plt.savefig('Error.png', format='png')


    plt.subplots()
    for vehicle in vehicles_list:
        plt.plot(vehicle.ht, vehicle.hv, label="vehicle[%d]'s speed"%vehicle.ID)
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [m/s]")
    plt.legend()
    plt.savefig('Spped.eps', format='eps')
    plt.savefig('Spped.png', format='png')

    plt.subplots()
    sns.ecdfplot(data=error_list, label="CDF of abs(error)")
    plt.grid(True)
    plt.xlabel("error [s]")
    plt.ylabel("CDF")
    plt.legend()
    plt.savefig('CDF.eps', format='eps')
    plt.savefig('CDF.png', format='png')

    plt.subplots()
    sns.histplot(data=error_list, kde=True, bins=200, label="Histogram & pdf of abs(error)")
    plt.grid(True)
    plt.xlabel("error [s]")
    plt.ylabel("Pdf")
    plt.legend()
    plt.savefig('pdf.eps', format='eps')
    plt.savefig('pdf.png', format='png')

    plt.show()

    percentile = np.percentile(error_list, 95)
    print(percentile)
    percentile_s = np.percentile(error_list_s, 95)
    print(percentile_s)

    with open('vehicles_list', 'wb') as f:
        pickle.dump(vehicles_list, f)

if __name__ == '__main__':
     main()
