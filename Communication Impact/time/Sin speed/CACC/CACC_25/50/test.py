import pickle
import matplotlib.pyplot as plt
from scipy.io import savemat

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
        self.v = 20
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
        self.y = self.y + self.v * np.sin(self.yaw) * DT
        self.yaw = self.yaw + self.v / self.WB * np.tan(delta) * DT
        self.v = self.v + a * DT

        if self.v > MAX_SPEED:
            self.v = MAX_SPEED
        elif self.v < MIN_SPEED:
            self.v = MIN_SPEED

        return self

    def receive(self, message_list):
        self.leaders = message_list

    # def error(self, error_list):
    #     self.errors = error_list


    def plot_car(self, cabcolor="-g", truckcolor="-k"):  # pragma: no cover

        outline = np.array([[-self.BACKTOWHEEL, (self.LENGTH - self.BACKTOWHEEL), (self.LENGTH - self.BACKTOWHEEL), -self.BACKTOWHEEL, -self.BACKTOWHEEL],
                            [self.WIDTH / 2, self.WIDTH / 2, - self.WIDTH / 2, -self.WIDTH / 2, self.WIDTH / 2]])

        fr_wheel = np.array([[self.WHEEL_LEN, -self.WHEEL_LEN, -self.WHEEL_LEN, self.WHEEL_LEN, self.WHEEL_LEN],
                             [-self.WHEEL_WIDTH - self.TREAD, -self.WHEEL_WIDTH - self.TREAD, self.WHEEL_WIDTH - self.TREAD, self.WHEEL_WIDTH - self.TREAD, -self.WHEEL_WIDTH - self.TREAD]])

        rr_wheel = np.copy(fr_wheel)

        fl_wheel = np.copy(fr_wheel)
        fl_wheel[1, :] *= -1
        rl_wheel = np.copy(rr_wheel)
        rl_wheel[1, :] *= -1

        Rot1 = np.array([[np.cos(self.yaw), np.sin(self.yaw)],
                         [-np.sin(self.yaw), np.cos(self.yaw)]])
        Rot2 = np.array([[np.cos(self.steer), np.sin(self.steer)],
                         [-np.sin(self.steer), np.cos(self.steer)]])

        fr_wheel = (fr_wheel.T.dot(Rot2)).T
        fl_wheel = (fl_wheel.T.dot(Rot2)).T
        fr_wheel[0, :] += self.WB
        fl_wheel[0, :] += self.WB

        fr_wheel = (fr_wheel.T.dot(Rot1)).T
        fl_wheel = (fl_wheel.T.dot(Rot1)).T

        outline = (outline.T.dot(Rot1)).T
        rr_wheel = (rr_wheel.T.dot(Rot1)).T
        rl_wheel = (rl_wheel.T.dot(Rot1)).T

        outline[0, :] += self.x
        outline[1, :] += self.y
        fr_wheel[0, :] += self.x
        fr_wheel[1, :] += self.y
        rr_wheel[0, :] += self.x
        rr_wheel[1, :] += self.y
        fl_wheel[0, :] += self.x
        fl_wheel[1, :] += self.y
        rl_wheel[0, :] += self.x
        rl_wheel[1, :] += self.y

        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), truckcolor)
        plt.plot(np.array(fr_wheel[0, :]).flatten(),
                 np.array(fr_wheel[1, :]).flatten(), truckcolor)
        plt.plot(np.array(rr_wheel[0, :]).flatten(),
                 np.array(rr_wheel[1, :]).flatten(), truckcolor)
        plt.plot(np.array(fl_wheel[0, :]).flatten(),
                 np.array(fl_wheel[1, :]).flatten(), truckcolor)
        plt.plot(np.array(rl_wheel[0, :]).flatten(),
                 np.array(rl_wheel[1, :]).flatten(), truckcolor)
        plt.plot(self.x, self.y, "*")

    def calc_ref_trajectory(self, cx, cy, cyaw, sp, dl):
        """
        calculating and updating reference trajectory and index of the path for the vehicle
        cx: course-x, cy: course-y, cyaw: speed profile, ck: curvature,
        sp: speed profile, dl: euclidean distance between waypoints
        """
        xref = np.zeros((NX, T + 1))
        dref = np.zeros((1, T + 1))
        ncourse = len(cx)
        pind = self.target_ind
        ind, _ = Agent.calc_nearest_index(self, cx, cy, cyaw, pind)

        if pind >= ind:
            ind = pind

        xref[0, 0] = cx[ind]
        xref[1, 0] = cy[ind]
        xref[2, 0] = sp[ind]
        xref[3, 0] = cyaw[ind]
        dref[0, 0] = 0.0  # steer operational point should be 0

        travel = 0.0

        for i in range(T + 1):
            travel += abs(self.v) * DT     # calculating index based on traversed local path
            dind = int(round(travel / dl))

            if (ind + dind) < ncourse:
                xref[0, i] = cx[ind + dind]
                xref[1, i] = cy[ind + dind]
                xref[2, i] = sp[ind + dind]
                xref[3, i] = cyaw[ind + dind]
                dref[0, i] = 0.0
            else:
                xref[0, i] = cx[ncourse - 1]
                xref[1, i] = cy[ncourse - 1]
                xref[2, i] = sp[ncourse - 1]
                xref[3, i] = cyaw[ncourse - 1]
                dref[0, i] = 0.0

        self.xref = xref
        self.target_ind =ind
        self.dref = dref

        return self

    def calc_nearest_index(self, cx, cy, cyaw, pind):
        """
        calculating nearest index of the path relative to the vehicle
        cx: course-x, cy: course-y, cyaw: speed profile,
        ck: curvature, sp: speed profile, dl: euclidean distance between waypoints
        """
        dx = [self.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
        dy = [self.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

        mind = min(d)

        ind = d.index(mind) + pind

        mind = np.sqrt(mind)

        dxl = cx[ind] - self.x
        dyl = cy[ind] - self.y

        angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
        if angle < 0:
            mind *= -1

        return ind, mind

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
                self.y = self.y + self.v * np.sin(self.yaw) * DT


def main():
    with open('vehicles_list', 'rb') as f:
        mylist = pickle.load(f)

    error_list = []
    i=1
    for vehicle in mylist:

        # a[i] = vehicle.ha
        # v[i] = vehicle.hv
        # e[i] = vehicle.error
        savemat('vehicle%d.mat'%i, {'A%d'%i: vehicle.ha, 'V%d'%i: vehicle.hv, 'E%d'%i:vehicle.error})
        i = i+1
        #plt.plot(vehicle.hx, vehicle.hy)#, label="vehicle[%d]'s trajectory"%vehicle.ID)

    # plt.grid(True)
    # plt.axis("equal")
    # plt.xlabel("x[m]")
    # plt.ylabel("y[m]")
    # plt.legend()

    #plt.show()

if __name__ == '__main__':
     main()
