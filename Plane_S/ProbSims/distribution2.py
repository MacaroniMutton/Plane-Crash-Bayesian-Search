# import numpy as np

# def makeGaussian(size, fwhm = 3, center=None):
#     """ Make a square gaussian kernel.

#     size is the length of a side of the square
#     fwhm is full-width-half-maximum, which
#     can be thought of as an effective radius.
#     """

#     x = np.arange(0, size, 1, float)
#     y = x[:,np.newaxis]

#     if center is None:
#         x0 = y0 = size // 2
#     else:
#         x0 = center[0]
#         y0 = center[1]

#     return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

# distr = makeGaussian(40)
# distr = distr / distr.sum()
# for row in distr:
#     for e in row:
#         print(f"{round(e*10,5)} ", end="")
#     print()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def makeGaussian(size, a, b, theta, center=None):

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    theta = (theta)*math.pi/180

    # return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    return np.exp(-4*np.log(2) * (((x-x0)*math.cos(theta)-(y-y0)*math.sin(theta))**2 / a**2 + ((y-y0)*math.cos(theta)+(x-x0)*math.sin(theta))**2 / b**2))
    

distr = makeGaussian(96, 7, 20, 90, center=[10, 10])
distr = distr / distr.sum()
# for row in distr:
#     for e in row:
#         print(f"{round(e*10,5)} ", end="")
#     print()
print(distr.max())
print(distr.mean())
print(distr.min())
hm = sns.heatmap(distr)
# print(hm.__dict__)
plt.show()
