import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
from scipy.integrate import odeint, quad

def deriv(X, t, lamda, mu):
    """returns the derivates dx1/dt and dx2/dt"""
    x1, x2 = X
    x1dot = mu*x1
    x2dot = lamda*(x2-x1**2)
    return x1dot, x2dot

def solve_equation(tmax, dt_per_period, x1_0, x2_0, lamda, mu):
    dt = 2*np.pi/dt_per_period
    t = np.arange(0,5, 0.02)
    X0 = [x1_0, x2_0]
    X = odeint(deriv, X0, t, args = (lamda, mu))
    return t, X
    
if __name__ == '__main__':
    x1_0, x2_0 = -5, 5
    tmax = 10
    lamda, mu = 1, -0.05
    dt_per_period = 100

    x1_data = []
    x2_data = []
    t_data = []

    for x1_0 in np.arange(-5, 0, 0.25):
        for x2_0 in np.arange(0, 5, 0.25):
            t, X = solve_equation(tmax, dt_per_period, x1_0, x2_0, lamda, mu)
            x1, x2 = X.T
            x1_data.append(x1)
            x2_data.append(x2)
            t_data.append(t)

    data = np.stack([np.array(x1_data), np.array(x2_data), np.array(t_data)], axis = 2)
    print(data.shape[1], data.shape[2])
    plt.plot(data[150][:,0], data[150][:,1])
    plt.show()
    np.save('equation.npy', data)