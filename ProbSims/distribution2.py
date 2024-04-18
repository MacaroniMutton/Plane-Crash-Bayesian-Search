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
from math import sin, cos, pi

def makeGaussian(size, fwhm = 4, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    theta = (110)*pi/180
    a = 30
    b = 10
    x0 = 20
    y0 = 48

    # return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    return np.exp(-4*np.log(2) * (((x-x0)*cos(theta)-(y-y0)*sin(theta))**2 / a**2 + ((y-y0)*cos(theta)+(x-x0)*sin(theta))**2 / b**2) )
    

distr = makeGaussian(96)
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
