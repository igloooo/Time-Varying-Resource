import numpy as np
from time import time


# adapted from  https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array

def shift5(arr, num, axis, fill_value=0):
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

BD = 40000
x = np.arange(0,BD**2).reshape(BD, BD)

# when BD = 4*10**4, time 32, 23, 25, 25s. The impact of transpose is negligible
time0 = time()
x_u  = shift5(x, -1, 0, 0)
print(time() - time0)

time0 = time()
x_l = shift5(x, 1, 0, 0)
print(time() - time0)

time0 = time()
x_le = shift5(x, -1, 1, 0)
print(time() - time0)

time0 = time()
x_ri = shift5(x, 1, 1, 0)
print(time() - time0)


# print(x)
# print(x_u)
# print(x_l)
# print(x_le)
# print(x_ri)

# y = x.reshape((2,2,2,2))
#
# print(y)
