from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.io import loadmat
import numpy as np

# ==============================================================================
# -- Trajectory Generation -----------------------------------------------------
# ==============================================================================
raw_data = loadmat('speed.mat')
speed = raw_data['Speed']
speed = np.array(speed)

t=2000

# ----------------------------------------------------------------------
# First the noiseless case
V_past = np.atleast_2d(np.linspace(t-4,t,5)).T

# Observations
temp_speed = speed[t-5:t]
V = temp_speed.ravel()

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
T = np.atleast_2d(np.linspace(t, t+100, 100)).T
real_speed = speed[t:t+100]

# Instantiate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor( kernel              = kernel,
                    alpha               = 0.01,
                    copy_X_train        = True,
                    optimizer           = "fmin_l_bfgs_b",
                    n_restarts_optimizer= 50)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(V_past, V)

# Make the prediction on the meshed x-axis (ask for MSE as well)
V_pred, sigma = gp.predict(T, return_std=True)
print(sigma)
# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure()
plt.plot(T, real_speed, 'r:', label=r'$Real-Speed$')
plt.plot(V_past, V, 'r.', markersize=10, label='Observations')
plt.plot(T, V_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([T, T[::-1]]),
         np.concatenate([V_pred - 1.9600 * sigma,
                        (V_pred + 1.9600 * sigma)[::-1]]),
         alpha=0.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$T$')
plt.ylabel('$Speed$')
plt.legend(loc='upper left')
plt.show()
