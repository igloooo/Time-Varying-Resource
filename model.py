
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix



class TwoStateModel:
    def __init__(self, M, mu0, mu1, mu, arr, p0, p1, BD, eps):
        # model parameters
        self.M = M
        self.mu0 = mu0
        self.mu1 = mu1
        self.mu = mu
        self.arr = arr
        self.p0 = p0
        self.p1 = p1
        self.arr0 = p0*arr
        self.arr1 = p1*arr
        self.y0, self.y1 = self.comp_job_amount(mu0, mu1, mu, arr, p0, p1)
        self.eps = eps
        # simulation specifications
        self.BD = BD
        # Q, Q_til are BD^2 x BD^2 matrices that apply to np.matmul(Q, x.reshape((-1,))).reshape((BD, BD))
        self.Q = self.comp_Q(mu0, mu1, mu, BD)
        self.Q_til = self.comp_Q_til(mu0, mu1, mu, BD)
        self.minus_e0_mat = self.comp_minus_e0_mat(BD)
        self.minus_e1_mat = self.comp_minus_e1_mat(BD)

        meshx, meshy = np.meshgrid(np.arange(0, BD), np.arange(0, BD), sparse=False, indexing='ij')
        self.h = np.clip(meshy - M, 0, None)
        self.h = self.h.reshape((-1,))

    def comp_job_amount(self, mu0, mu1, mu, arr, p0, p1):
        y0 = None
        y1 = None
        return y0, y1

    def comp_Q(self, mu0, mu1, mu, BD):
        # Q[m, n, i, j] = rate from (k0, k1) = (i, j) to (m, n)
        Q_mat = np.zeros((BD, BD, BD, BD))
        for i in range(BD):
            for j in range(BD):
                # rate leaving state (i, j)
                Q_mat[i, j, i, j] = -(mu0 * i + mu1 * j + mu * (i+j))
                if i >= 1:
                    if j < BD-1:############leak
                        # rate of 0->1 event
                        Q_mat[i-1, j+1, i, j] = mu0 * i
                    else:
                        Q_mat[i, j, i, j] += mu * i
                    # rate of 0->* event
                    Q_mat[i-1, j, i, j] = mu * i
                if j >= 1:
                    if i < BD - 1:
                        # rate of 1->0 event
                        Q_mat[i+1, j-1, i, j] = mu1 * j
                    else:
                        Q_mat[i, j, i, j] += mu * i
                    # rate of 1->* event
                    Q_mat[i, j-1, i, j] = mu * j

        Q_mat = Q_mat.reshape((BD*BD, BD*BD))
        Q_mat_csc = csc_matrix(Q_mat)
        return Q_mat_csc

    def comp_Q_til(self, mu0, mu1, mu, BD):
        Q_til_mat = np.zeros((BD, BD, BD, BD))
        for i in range(BD):
            for j in range(BD):
                # rate leaving state (i, j)
                Q_til_mat[i, j, i, j] = -mu * (i+j)
                if i >= 1:
                    # rate of 0->* event
                    Q_til_mat[i-1, j, i, j] = mu * i
                if j >= 0:
                    # rate of 1->* event
                    Q_til_mat[i, j-1, i, j] = mu * j

        Q_til_mat = Q_til_mat.reshape((BD*BD, BD*BD))
        Q_til_mat_csc = csc_matrix(Q_til_mat)
        return Q_til_mat_csc

    def comp_minus_e0_mat(self, BD):
        #leak!
        mat = np.zeros((BD, BD, BD, BD))
        for i in range(1, BD):
            for j in range(BD):
                mat[i, j, i-1, j] = 1
                if i == BD-1:
                    mat[i, j, i, j] = 1

        mat = mat.reshape((BD*BD, BD*BD))
        mat_sp = csr_matrix(mat)
        return mat_sp

    def comp_minus_e1_mat(self, BD):
        #leak!
        mat = np.zeros((BD, BD, BD, BD))
        for i in range(BD):
            for j in range(1, BD):
                mat[i, j, i, j-1] = 1
                if j == BD-1:
                    mat[i, j, i, j] = 1

        mat = mat.reshape((BD*BD, BD*BD))
        mat_sp = csr_matrix(mat)
        return mat_sp


def shift_arr(arr, num, axis, fill_value=0):
    """
    adapted from  https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
    :param arr: the array to be shifted; usually expect two dimension
    :param num: displacement, positive means shift downward or rightward, depending on axis
    :param axis: shift along which axis axis=0 means shift vertically;
                shift axis=1 means shift horizontally, and so on
    :param fill_value: fill value
    :return:
    """
    if axis != 0:
        n_dims = len(arr.shape)
        dims = list(range(n_dims))
        dims.remove(axis)
        dims.insert(0, axis)
        arr = arr.transpose(dims)

    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr

    if axis != 0:
        n_dims = len(arr.shape)
        dims = list(range(n_dims))
        dims.remove(0)
        dims.insert(axis, 0)
        result = result.transpose(dims)
    return result


