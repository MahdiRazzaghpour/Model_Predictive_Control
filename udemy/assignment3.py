import numpy as np
from sim.sim2d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['OBSTACLES'] = True

class ModelPredictiveControl:
    def __init__(self):
        self.horizon = 10
        self.dt = 0.2

        # Reference or set point the controller will achieve.
        self.reference1 = [10, 0, 0]
        self.reference2 = None

        self.x_obs = 5
        self.y_obs = 0.1

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
            state = self.plant_model(state, self.dt, u[k*2], u[k*2+1])
            # position cost
            cost += abs(ref[0]-state[0])**2 *10
            cost += abs(ref[1]-state[1])**2 *10
            # Angle cost
            cost += abs(ref[2]-state[2])**2
            # steering cost
            cost += self.obstacle_cost(state[0],state[1]) *10
        return cost

    def obstacle_cost(self, x , y):
        distance=np.sqrt((x-self.x_obs)**2+(y-self.y_obs)**2)
        if (distance > 3):
            return 0
        else:
            return 1/distance*30
sim_run(options, ModelPredictiveControl)
