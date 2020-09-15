import numpy as np
from sim.sim2d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['OBSTACLES'] = False

class ModelPredictiveControl:
    def __init__(self):
        self.horizon = 15
        self.dt = 0.2

        # Reference or set point the controller will achieve.
        self.reference1 = [10, 10, 0]
        self.reference2 = None# [10, 2, 3.14/2]

    def plant_model(self,prev_state, dt, pedal, steering):
        x_t = prev_state[0]
        y_t = prev_state[1]
        psi_t = prev_state[2]
        v_t = prev_state[3]

        beta = steering
        a_t = pedal

        x_t = x_t + v_t*np.cos(psi_t)*dt
        y_t = y_t + v_t*np.sin(psi_t)*dt
        psi_t = psi_t + v_t*np.tan(beta)/2.5
        v_t = v_t + a_t*dt- v_t/25

        return [x_t, y_t, psi_t, v_t]

    def cost_function(self,u, *args):
        state = args[0]
        ref = args[1]
        cost = 0.0
        for k in range(0, self.horizon):
            ts = [0,1]
            v_start = state[3]
            state = self.plant_model(state, self.dt, u[k*2], u[k*2+1])

            # position cost
            cost += abs(ref[0]-state[0])**2
            cost += abs(ref[1]-state[1])**2
            # Angle cost
            cost += abs(ref[2]-state[2])
            # steering cost
            cost += u[k*2+1]**2


        return cost

sim_run(options, ModelPredictiveControl)
