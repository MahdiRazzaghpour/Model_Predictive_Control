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
V_N = 3 #number of vehicles
inter_vehicle_time = 0.8 #s
communication_error_rate = 0.0 # percent

N = 10    # prediction horizon

CT = 0.1    # communication time tick [s]
t_s  = 0.1  # simulator time tick [s]
dl = 1.0  # course tick [m]
NX = 4    # x = [x, y, v, yaw]
NU = 2    # a = [accel, steer]

u_min = -4
u_max = 4
V_max = 30         # maximum speed [m/s]
V_min = -10        # minimum speed [m/s]
a_max = 3          # maximum accel [m/ss]
a_min = -4

T = 1
l_v = 5
f = 0.1
l = 2+l_v
# ==============================================================================
# --  MPC parameters  ----------------------------------------------------------
# ==============================================================================
R = np.array([l,0,0])
Q = np.array([ [2,0,0] , [0,1,0] , [0,0,1] ])

A = np.array([ [1 ,t_s,-T*t_s] , [0, 1 ,-t_s] , [0,0,1-t_s/f] ])
B = np.array([0 , 0 , t_s/f])
D = np.array([0 , t_s , 0])

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
sp = raw_data['V']
# sp=sp[0:100,:]
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

        self.leaders = []
        self.error = 0
        self.error_bias=0


class MPC:
    """
    MPC control class, solving optimization problem over cvxpy optimizer
    with respect to cost functions and constraints
    """
    def __init__(self):
        self.horizon = N    # prediction horizon
        self.t_s = t_s

    # leader MPC function
    def SHMPCL (A,B,D,acc,x_0,N,R,Q,l,T,V_0,t_s):
        x = Variable((3,N+1))
        u = Variable((N))
        v = Variable((N+1))
        ctrs = [x[:,0]==x_0 , v[0]==V_0]  # initial point
        objective = cp.quad_form(x[:,N]-R,Q)  # terminal cost
        ctrs += [x[2,:]>=a_min , x[2,:]<=a_max]
        ctrs += [u[:]>=u_min , u[:]<=u_max]
        ctrs += [ v[:]<=35 , v[:]>=0 ]
        for k in range(N):
            ctrs += [x[:,k+1]==A@x[:,k]+B*u[k]]
            ctrs += [v[k+1]==v[k]+x[2,k]*t_s]
            objective += cp.quad_form(x[:,k]-R,Q)

        objective_cp=cp.Minimize(objective)
        prob = cp.Problem(objective_cp,ctrs)
        prob.solve(verbose=False,solver=cp.GUROBI)
        np.where(v.value[:]<=0.0001,0,x.value[2,:])
        return x.value,v.value

    # follower MPC function
    def SHMPCF (A,B,D,acc,x_0,N,R,Q,l,T,V_0,t_s,P_w,P_hat,q_p,i):
        x = Variable((3,N+1))
        u = Variable((N))
        v = Variable((N+1))
        d2 = Variable(N,boolean=True)
        f_1 = Variable(N,boolean=True)

        ctrs = [x[:,0]==x_0 , v[0]==V_0]  # initial point

        objective = cp.quad_form(x[:,N]-R,Q)  # terminal cost
        chance_ctr = 0
        ctrs += [x[2,:]>=a_min , x[2,:]<=a_max]
        ctrs += [u[:]>=u_min , u[:]<=u_max]
        ctrs += [v[:]<=35 , v[:]>=0] # speed limit
        for k in range(N):
            ctrs += [x[:,k+1]==A@x[:,k]+acc[k]*D + B*u[k] ]
            ctrs += [v[k+1]==v[k]+x[2,k]*t_s]
            ctrs += [v[k]-(2*abs(a_min)*t_s)<=100*f_1[k] , v[k]-(2*abs(a_min)*t_s)>=2**-50 + (1-f_1[k])*(-100-2**-50)]

            ctrs += [x[0,k]>=5.0001-v[k]*T]
            ###################################################################
            #################### emergency braking condition
            ctrs += [x[0,k]-l+1<=200*(1-d2[k])]
            ctrs += [x[0,k]-l+1>=2**-50 + d2[k]*(-200-2**-50)]
            ctrs += [u[k]<=(d2[k]+f_1[k])/2*(u_min) + (2 - (d2[k]+f_1[k]))*10]
            ###################################################################
            objective += cp.quad_form(x[:,k]-R,Q)
        objective_cp=cp.Minimize(objective)
        prob = cp.Problem(objective_cp,ctrs)
        prob.solve(verbose=False,solver=cp.GUROBI)
        if int(d2.value[0]) == 1:
            print(f'EMS BRAKE for vehicle {i}' )
        return x.value,v.value,d2.value[0]


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
        print(counter)

        for vehicle in vehicles_list:
            if (counter % 1) == 0 :
                vehicle.receive(message_list)

            oa, odelta, ox, oy, oyaw, ov = MPC.iterative_linear_mpc_control(vehicle.xref, vehicle, vehicle.dref, vehicle.oa, vehicle.odelta,vehicles_list)


            ###################################################################
            if i==0: # leader vehicle
                acc = np.zeros((N+1))
                A = np.array([ [0,0,0] , [0, 1 ,t_s] , [0,0,1-t_s/f] ])
                V_max = 20
                V_min = -20
                Q = np.array([ [0,0,0] , [0,5,0] , [0,0,1] ])
                w_0 = np.zeros(3)
                if t<220:  # different speed references for the leader vehicle  150
                    R = np.array([0,27,0])
                elif t<400: #250
                    R = np.array([0,0,0])
                else:
                    R = np.array([0,20,0])

                x_0 = np.array([q_t[0,t],v_t[0,t],a_t[0,t]])
            ####################################################################
            else: # follower vehicles
                A = np.array([ [1 ,t_s,-T*t_s] , [0, 1 ,-t_s] , [0,0,1-t_s/f] ])
                acc = a[i-1,:]
                R = np.array([l,0,0])
                V_min = -20
                V_max = 20
                Q = np.array([ [2,0,0] , [0,1,0] , [0,0,1] ])
                r = random.random()
                w_0 = np.zeros(np.size(P_w))
                w_0[np.where(r<=AP_w)[0][0]] = 1
                E[i,t] = M_w@w_0
                x_0 = np.array([dd_t[i,t]+M_w@w_0,d_v[i],a_t[i,t]])

            ########### calling the MPC function
            if i==0:  # MPC for the leader
                O = SHMPCL (A,B,D,acc,V_min,V_max,a_min,a_max,x_0,N,R,Q,l,T,v_t[i,t],t_s,u_min,u_max)
            else:     # MPC for the followers
                #tt = time.process_time()
                O = SHMPCF (A,B,D,acc,V_min,V_max,a_min,a_max,x_0,N,R,Q,l,T,v_t[i,t],
                                t_s,np.log(P_w),np.log(P_hat),q_p,i,u_min,u_max,G_w)
                #elapsed = time.process_time() - tt
                #print(elapsed)
            ###########################################################################
            d_v[i] = O[0][1,1]
            v_t[i,t+1] = O[1][1]
            if v_t[i,t+1]<0:
                v_t[i,t+1]=0
            a_t[i,t+1] = O[0][2,1]
            q_t[i,t+1] = q_t[i,t] + t_s*v_t[i,t]
            r = random.random()
            if r>0.9:
                loss_flag[i,t]=1
            #d_v2[i] = v_t[i-1,t+1]-v_t[i,t+1]
            if t%comm_rate == 0 and loss_flag[i,t]==0:  # data exchange every 5*t_s
                aa[i,:] = O[0][2,:]
                d_v2[i] = v_t[i-1,t+1]-v_t[i,t+1]
            elif i>0:
                d_v2[i] = O[0][1,1]
                a[i,0] = (a[i,-1]-a[i,-2]) + a[i,-1]

                if a[i,0]>a_max:
                    a[i,0]=a_max
                elif a[i,0]<a_min:
                    a[i,0]=a_min
                aa[i,:] = np.roll(a[i,:],-1)

            if i==0:
                dd_t[i,t+1] = O[0][0,1]
            else:
                dd_t[i,t+1] = q_t[i-1,t+1] - q_t[i,t+1] - T*v_t[i,t+1]
                d_e[i,t] = O[2]


            vehicle.update_state(ai, di)
            vehicle.hx = np.append(vehicle.x, vehicle.hx)
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

        time = time + t_s
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
