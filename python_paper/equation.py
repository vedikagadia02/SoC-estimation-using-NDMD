import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
from scipy.integrate import odeint, quad, RK45, solve_ivp

def deriv(t, X, lamda, mu):
    """returns the derivates dx1/dt and dx2/dt"""
    x1, x2 = X
    x1dot = -0.1*x1
    x2dot = -1*(x2-x1**2)
    # x2dot = lamda*x2
    return x1dot, x2dot

# def solve_equation(tmax, dt_per_period, x1_0, x2_0, lamda, mu): trying out new settings
def solve_equation(tmax, dt, x1_0, x2_0, lamda, mu):
    # dt = 2*np.pi/dt_per_period
    # t = np.arange(0,5, 0.02) trying out new settings
    t = np.arange(0,50, 0.1)
    X0 = [x1_0, x2_0]
    X = solve_ivp(deriv, [t[0], t[-1]], X0, method='RK45', t_eval=t, args=(lamda, mu))
    return t, X.y

if __name__ == '__main__':
    # x1_0, x2_0 = -2,-2 trying out new settings
    # tmax = 10 trying out new settings
    tmax = 2
    # lamda, mu = 1, 0.05 trying out new settings
    lamda, mu = -1, -0.1
    # dt_per_period = 100 trying out new settings
    dt = 0.1

    x1_data = []
    x2_data = []
    t_data = []

    for x1_0 in np.arange(-2, 2, 0.1):
        for x2_0 in np.arange(-2, 2, 0.1):
            # t, X = solve_equation(tmax, dt_per_period, x1_0, x2_0, lamda, mu) trying out new settings
            t, X = solve_equation(tmax, dt, x1_0, x2_0, lamda, mu)
            x1, x2 = X
            x1_data.append(x1)
            x2_data.append(x2)
            t_data.append(t)

    data = np.stack([np.array(x1_data), np.array(x2_data), np.array(t_data)], axis = 2)
    print(data.shape[0], data.shape[1], data.shape[2])
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Create a figure with 3 subplots

    # Plot x1_data vs t_data
    for i in range(len(t_data)):
        axs[0].plot(t_data[i], x1_data[i])
    axs[0].set_title('x1_data vs t_data')
    axs[0].set_xlabel('t_data')
    axs[0].set_ylabel('x1_data')

    # Plot x2_data vs t_data
    for i in range(len(t_data)):
        axs[1].plot(t_data[i], x2_data[i])
    axs[1].set_title('x2_data vs t_data')
    axs[1].set_xlabel('t_data')
    axs[1].set_ylabel('x2_data')

    # Plot x1_data vs x2_data
    for i in range(len(x1_data)):
        axs[2].plot(x1_data[i], x2_data[i])
    axs[2].set_title('x1_data vs x2_data')
    axs[2].set_xlabel('x1_data')
    axs[2].set_ylabel('x2_data')

    plt.tight_layout()
    plt.show()
    np.save('equation.npy', data)