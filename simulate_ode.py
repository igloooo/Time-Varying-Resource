"""
Simulate job migration model with a single type of
two-state markov chain jobs at a fluid scale
"""

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.animation as animation
import matplotlib
from functools import partial
from scipy.sparse import csc_matrix
from scipy.integrate import solve_ivp
from collections import namedtuple
from model import TwoStateModel, shift_arr




def grandaz_ktilde(x, x_til, arr0, arr1, az, k_til):
    """
    a policy class similar to GRAND(az) and acts on tilde system
    :param x: BDxBD array, the current state
    :param x_til: BDxBD array, the current state in the tilde system
    :param arr0, arr1: the arrival rate of two types of jobs
    :param az: the amount of empty servers
    :param k_til: a BDxBD 0-1 array denoting the set that we rout from
    :return: two BDxBD array, the rate of allocating the jobs
            to servers at each states, must sum up to arr0 and arr 1
    """
    # assume fictionally az empty servers
    x_til[0,0] = az
    x[0,0] = az
    zero_job_weights = x_til * shift_arr(k_til, -1, 0, 0)
    one_job_weights = x_til * shift_arr(k_til, -1, 1, 0)
    v0 = arr0 / np.sum(zero_job_weights) * zero_job_weights
    v1 = arr1 / np.sum(one_job_weights) * one_job_weights
    return v0, v1

def grandaz_k(x, x_til, arr0, arr1, az, k_til):
    """
    a policy class similar to GRAND(az) and acts on the original system
    :param x: BDxBD array, the current state
    :param x_til: BDxBD array, the current state in the tilde system
    :param arr0, arr1: the arrival rate of two types of jobs
    :param az: the amount of empty servers
    :param k_til: a BDxBD 0-1 array denoting the set that we rout from
    :return: two BDxBD array, the rate of allocating the jobs
            to servers at each states, must sum up to arr0 and arr 1
    """
    # assume fictionally az empty servers
    x_til[0,0] = az
    x[0,0] = az
    zero_job_weights = x * shift_arr(k_til, -1, 0, 0)
    one_job_weights = x * shift_arr(k_til, -1, 1, 0)
    v0 = arr0 / np.sum(zero_job_weights) * zero_job_weights
    v1 = arr1 / np.sum(one_job_weights) * one_job_weights
    return v0, v1



# define the dynamics, partial function used as input to ode solver
def dynamics(t, x_n_xtil, model, policy_partial):
    """
    :param x_n_xtil: (BDxBDx2,) vector, x_n_xtil[:BD*BD] is x, x_n_xtil[BD*BD:] is x_til
    :param policy: the specific policy to simulate, a partial function with parameters specified
    :param model: the set of parameters
    :return: deriv of x and x_til, also in the form of a 2xBDxBD array
    """
    BD = model.BD

    deriv = np.zeros((2*BD*BD,))
    x = x_n_xtil[:BD*BD]
    x_til = x_n_xtil[BD*BD:]
    # set number of empty servers to 0
    x[0] = 0
    x_til[0] = 0
    # change triggered by mutation
    deriv[:BD*BD] = model.Q.dot(x)
    deriv[BD*BD:] = model.Q_til.dot(x_til)
    # change triggered by arrival
    arr_change = np.zeros((BD*BD,))
    v0, v1 = policy_partial(x.reshape((BD, BD)), x_til.reshape((BD, BD)))
    arr_change -= (v0+v1).reshape((-1,))
    arr_change += model.minus_e0_mat.dot(v0.reshape((-1,)))  #shift_arr(v0, num=1, axis=0, fill_value=0).reshape((-1,))
    arr_change += model.minus_e1_mat.dot(v1.reshape((-1,))) #shift_arr(v1, num=1, axis=1, fill_value=0).reshape((-1,))
    deriv[:BD*BD] += arr_change
    deriv[BD*BD:] += arr_change
    # not change amount of empty servers
    deriv[0] = 0
    deriv[BD*BD] = 0

    # print('x=', x.reshape((BD, BD)))
    # print('dx=', deriv[0:BD**2].reshape((BD, BD)))

    return deriv


def update_plot(frame_number, meshx, meshy, zarray, plot):
    """
    taken from https://pythonmatplotlibtips.blogspot.com/2018/11/animation-3d-surface-plot-funcanimation-matplotlib.html
    :param frame_number:
    :param zarray:
    :param plot:
    :return:
    """
    plot[0].remove()
    plot[0] = ax.plot_surface(meshx, meshy, zarray[:,:,frame_number], cmap="magma")


if __name__ == '__main__':
    my_model = TwoStateModel(
        M=5,
        mu0=10,
        mu1 = 20,
        mu = 0.5,
        arr = 1000,
        p0 = 0.4,
        p1 = 0.6,
        BD = 10,
        eps = 0.01
    )

    az = 0.01
    k_max = 15
    based_on = 'original' #'tilde' # or 'original'

    meshx, meshy = np.meshgrid(np.arange(0, my_model.BD), np.arange(0, my_model.BD), sparse=False, indexing='ij')
    k_til = (meshx+3*meshy) <= k_max

    T = 500
    frn = 50

    t_span = np.linspace(0, T, frn)

    fps = 25

    exp_name = '13_lower_freq_k'

    # system state x, and x_til
    # the i-th axis correspond to k_i; if we draw a matrix, k_0 is along the vertical axis
    x = np.zeros((my_model.BD * my_model.BD, ))
    x_til = np.zeros((my_model.BD * my_model.BD, ))
    initial_state = np.concatenate((x, x_til))

    if based_on == 'original':
        policy_partial = partial(grandaz_k,
                         arr0 = my_model.arr0,
                         arr1=my_model.arr1,
                         az=az,
                         k_til=k_til
                         )
    elif based_on == 'tilde':
        policy_partial = partial(grandaz_ktilde,
                         arr0 = my_model.arr0,
                         arr1 = my_model.arr1,
                         az=az,
                         k_til=k_til
                         )
    else:
        raise ValueError

    dynamics_cur = partial(dynamics, model=my_model, policy_partial=policy_partial)
    #deriv = dynamics_cur(1, initial_state)
    #print(deriv[:40*40].reshape((40,40)), deriv[40*40:].reshape((40,40)))

    sol = solve_ivp(dynamics_cur, t_span, initial_state)
    zarray = sol.y[:my_model.BD**2,:].reshape((my_model.BD, my_model.BD, -1))
    zarray[0,0,:]=0

    # print(np.sum(z > 0))
    # print(np.sum(k_til))


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plot = [ax.plot_surface(meshx, meshy, zarray[:,:,0], color='0.75', rstride=1, cstride=1)]
    #ax.view_init(azim=0, elev=90)
    ax.set_xlabel('k_0')
    ax.set_ylabel('k_1')
    zlim = 1#np.max(zarray)
    ax.set_zlim(0,zlim)
    ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(meshx, meshy, zarray, plot), interval=1000/fps)

    fn = 'gifs/'+ exp_name
    writergif = animation.PillowWriter(fps=fps)
    #ani.save(fn+'.mp4',writer='ffmpeg',fps=fps)
    ani.save(fn+'.gif',writer=writergif)
    #ani.save(fn+'.gif', writer='imagemagick',fps=fps)
