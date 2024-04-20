import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal


# Set the mean and covariance matrix for the bivariate normal distribution
mean = np.array([0, 0])
covariance_matrix = np.array([[1, 0.5], [0.5, 1]])

# Create a grid of x and y values
x = np.linspace(-3, 3, 20)
y = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Create the bivariate normal distribution
rv = multivariate_normal(mean, covariance_matrix)

# Calculate the PDF values for each point in the grid
Z = rv.pdf(pos)
for i in range(len(Z)):
    for j in range(len(Z[0])):
        Z[i][j] *= 10000
print(Z)