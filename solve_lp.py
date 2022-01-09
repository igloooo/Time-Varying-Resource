import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp
import numpy as np
from model import TwoStateModel, shift_arr

import matplotlib.pyplot as plt
from matplotlib import cm


my_model = TwoStateModel(
        M=10,
        mu0=1,
        mu1 = 2,
        mu = 10,
        arr = 1000,
        p0 = 0.4,
        p1 = 0.6,
        BD = 20,
        eps = 0.01
    )

m = gp.Model('opt_control')

BD = my_model.BD

x = m.addMVar((BD**2,), lb=0, vtype=GRB.CONTINUOUS, name='x')
v0 = m.addMVar((BD**2,), lb=0, vtype=GRB.CONTINUOUS, name='v0')
v1 = m.addMVar((BD**2,), lb=0, vtype=GRB.CONTINUOUS, name='v1')


Q = my_model.Q
Me0 = my_model.minus_e0_mat
Me1 = my_model.minus_e1_mat

m.setObjective(x[1:].sum(), GRB.MINIMIZE)

m.addConstr(Q @ x + Me0 @ v0 + Me1 @ v1 - (v0+v1) == 0, name='stationary')


h = my_model.h
eps = my_model.eps
m.addConstr(h @ x <= eps * x[1:].sum(), name='capacity')

m.addConstr(h @ v0 == 0, name='reasonable0')
m.addConstr(h @ v1 == 0, name='reasonable1')

arr0 = my_model.arr0
arr1 = my_model.arr1
m.addConstr(v0.sum() == arr0, name='arrival0')
m.addConstr(v1.sum() == arr1, name='arrival1')

m.addConstrs((v0[i] * my_model.maskv[i] == 0 for i in range(len(my_model.maskv))), name='boundv0')
m.addConstrs((v1[i] * my_model.maskv[i] == 0 for i in range(len(my_model.maskv))), name='boundv1')
m.addConstrs((x[i] * my_model.maskx[i] == 0 for i in range(len(my_model.maskx))), name='boundx')




m.optimize()


meshx, meshy = np.meshgrid(np.arange(0, BD), np.arange(0, BD), sparse=False, indexing='ij')
z = x.X.reshape((BD, BD))
z[0,0] = 0
v0_val = v0.X.reshape((BD, BD))
v1_val = v1.X.reshape((BD, BD))

#print(z)
# print(v0_val)
# print(v1_val)
print('eps=', eps)
print('lhs/sumx=', h.dot(x.X) / np.sum(x.X))
print(x.X[0])


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
plot = ax.plot_surface(meshx, meshy, v0_val, color='0.75', rstride=1, cstride=1)
ax.set_xlabel('k0')
ax.set_ylabel('k1')
plt.show()






