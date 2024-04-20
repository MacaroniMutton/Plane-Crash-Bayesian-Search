import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

mean = [0, 0]
std_dev = 1

# Create a grid of polar coordinates
theta = np.linspace(0, 2 * np.pi, 100)
r = np.ones_like(theta)  # Constant radius for a circular distribution

# Convert polar coordinates to Cartesian coordinates
x = r * np.cos(theta)
y = r * np.sin(theta)

# Create the circular normal distribution
rv = multivariate_normal(mean, std_dev)

# Calculate the PDF values for each point in the grid
pdf_values = rv.pdf(np.column_stack((x, y)))
print(pdf_values)