import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import cvxpy as cp
from cvxpy import *
import random
import scipy.stats as ss
import time

# leader MPC function
def SHMPCL (A,B,D,acc,x_0,R,Q,l,V_0):
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

# # follower MPC function
def SHMPCF (A,B,D,acc,x_0,R,Q,l,V_0,q_p,i):
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

    for j in range(V_N):
        if (j - V_NC >= 0):
            for k in range(N):
                ctrs += [x[:,k+1]==A@x[:,k]+acc[j,k]*D + B*u[k]]
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

# Main
V_N = 5 #number of vehicles
V_NC = 4
N = 2  #prediction horizon
T = 1

T_F = 700
t_s = 0.1 # sampling time
l_v = 5
f = 0.1
l = 2+l_v
V_min = -20
V_max = 20
a_min = -4
a_max = 3
u_min = -4
u_max = 4
A = np.array([ [1 ,t_s,-T*t_s] , [0, 1 ,-t_s] , [0,0,1-t_s/f] ])
B = np.array([0 , 0 , t_s/f])
D = np.array([0 , t_s , 0])

q_p = 100

####### independent dynamics
q_t = np.zeros((V_N,T_F+1))
v_t = np.zeros((V_N,T_F+1))
a_t = np.zeros((V_N,T_F+1))
####### string dynamics
dd_t = np.zeros((V_N,T_F+1))
############################
x_0 = np.zeros((V_N,3))
acc = np.zeros((V_N,N+1))
a = np.zeros((V_N,N+1))
aa = np.zeros((V_N,N+1))
d_e = np.zeros((V_N,T_F+1))
####### initial values
q_t[:,0] = 23*np.arange(V_N,0,-1)
v_t[:,0] = 15*np.ones(V_N)
a_t[:,0] = np.zeros(V_N)
d_v = np.zeros(V_N)
d_v2 = np.zeros(V_N)
loss_flag = np.zeros((V_N,T_F))
E = np.zeros((V_N,T_F))
comm_rate = 10  # communicate every comm_rate sample

for j in range(1,V_N):
    d_v2[j] = v_t[j-1,0]-v_t[j,0]
    dd_t[j,0]= q_t[j-1,0]-q_t[j,0]- T*v_t[j,0]  ## dd_t[0,:] is meaningless

for t in range(T_F):
    print (t)
    a = aa
    d_v = d_v2
    for i in range (V_N):
        ###################################################################
        if i==0: # leader vehicle
            acc = np.zeros((V_N,N+1))
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
            acc[i,:] = a[i-1,:]
            R = np.array([l,0,0])
            V_min = -20
            V_max = 20
            Q = np.array([ [2,0,0] , [0,1,0] , [0,0,1] ])
            x_0 = np.array([dd_t[i,t],d_v[i],a_t[i,t]])

        ########### calling the MPC function
        if i==0:  # MPC for the leader
            O = SHMPCL (A,B,D,acc,x_0,R,Q,l,v_t[i,t])
        else:     # MPC for the followers
            #tt = time.process_time()
            O = SHMPCF (A,B,D,acc,x_0,R,Q,l,v_t[i,t],q_p,i)
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
        if r>0.99:
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

# Plot
# Plotting
SMALL_SIZE = 32
MEDIUM_SIZE = 24
BIGGER_SIZE = 24
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

abcd = [':', '-.' , '+','-','*','--','x','3-','|-']
color1 = ['b' , 'g' , 'r' , 'mediumaquamarine' , 'deepskyblue' , 'tab:orange' , 'darkblue' , 'tab:purple' , 'darkred']
x1 = []
for j in range(T_F+1):
    x1.append(j*t_s)
color=cm.rainbow(np.linspace(0,1,V_N-1))

fig, axs = plt.subplots(4,gridspec_kw={'height_ratios': [1, 1,1,0.3]})
fig.set_size_inches(28, 33)
#fig.set_size_inches(27, 33)
###########################################################################
counter12 = 0
for i,c in zip(range(V_N-1),color):
    s='$d_{%i}$' % (i+1)
    axs[0].plot(x1,q_t[i,:]-q_t[i+1,:]-l_v,abcd[counter12],c=color1[counter12],label=s)
    s='$\Delta d_{%i}$' % (i+1)
    axs[0].plot(x1,q_t[i,:]-q_t[i+1,:]-T*v_t[i+1,:]-l,abcd[counter12],c=color1[counter12],label=s)
    counter12 += 1
    if counter12>8:
        counter12 = 0
axs[0].set(xlabel='time (s)', ylabel='distance (m)')
axs[0].set_xlim(x1[0],x1[-1])
axs[0].set_ylim(-5,45)
axs[0].grid()
axs[0].legend(ncol=9,handleheight=2,loc='upper center')
axs[0].set_title('regulated distance',fontsize='large',fontweight='bold')
#############################################################################
counter12 = 0
for i,c in zip(range(1,V_N),color):
    s='$v_{%i}$' % (i)
    axs[1].plot(x1,v_t[i,:],abcd[counter12],c=color1[counter12],label=s)
    s='$\Delta v_{%i}$'% (i)
    axs[1].plot(x1,v_t[i-1,:]-v_t[i,:],abcd[counter12],c=color1[counter12],label=s)
    counter12 += 1
    if counter12>8:
        counter12 = 0
axs[1].plot(x1,v_t[0,:],'k-.',linewidth=2,label='$v_{0}$')
axs[1].set(xlabel='time (s)', ylabel='velocity (m/s)')
axs[1].set_xlim(x1[0],x1[-1])
axs[1].set_ylim(-5,40)
axs[1].grid()
axs[1].legend(ncol=10,handleheight=2,loc='upper center')
axs[1].set_title('regulated velocity',fontsize='large',fontweight='bold')
#############################################################################
axs[2].plot(x1,a_t[0,:],'k-.',linewidth=2,label='$a_{0}$')
counter12 = 0
for i,c in zip(range(1,V_N),color):
    s='$a_{%i}$'%(i)
    axs[2].plot(x1,a_t[i,:],abcd[counter12],c=color1[counter12],label=s)
    counter12 += 1
    if counter12>8:
        counter12 = 0
axs[2].set(xlabel='time (s)', ylabel='acceleration $(m/s^{2})$')
axs[2].set_xlim(x1[0],x1[-1])
axs[2].set_ylim(-4.5,5)
axs[2].grid()
axs[2].legend(ncol=V_N,handleheight=2,loc='upper center')
axs[2].set_title('regulated acceleration',fontsize='large',fontweight='bold')
##############################################################################
counter12 = 0
for i,c in zip(range(1,V_N),color):
    s='$\delta^{e}_{%i}$'%(i)
    axs[3].plot(x1,d_e[i,:],abcd[counter12],c=color1[counter12],label=s)
    counter12 += 1
    if counter12>8:
        counter12 = 0
axs[3].set(xlabel='time (s)', ylabel='$\delta^{e}$')
axs[3].set_xlim(x1[0],x1[-1])
axs[3].set_ylim(-0.25,2.85)
axs[3].set_yticks([0,1])
axs[3].grid()
axs[3].legend(ncol=V_N-1,handleheight=2,loc='upper right')
axs[3].set_title('emergency braking status',fontsize='large',fontweight='bold')
plt.subplots_adjust(hspace=0.3)
fig.savefig(r'C:\Users\ma121036\Documents\GitHub\Model_Predictive_Control\SMPC\DHSA_tc_10ts_loss.jpg', dpi=200,bbox_inches = 'tight',pad_inches = 0)
