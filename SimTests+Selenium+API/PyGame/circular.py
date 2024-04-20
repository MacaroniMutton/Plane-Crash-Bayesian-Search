import numpy as np

# Define parameters
num_points = 100  # Number of points in the distribution
theta_values = np.linspace(0, 2 * np.pi, num_points)  # Angles from 0 to 2*pi
radius = 1  # Radius of the circular distribution

# Create the 2D array representing the circular probability distribution
circular_distribution = np.cos(theta_values)

# Scale the distribution to ensure non-negative values
circular_distribution = circular_distribution / np.max(circular_distribution)

# Print the resulting array
print(circular_distribution)
