import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from sklearn.neighbors import KernelDensity

# Define grid parameters
grid_resolution = (100, 100)  # Grid resolution (number of cells)
grid_extent = ((0, 10), (0, 10))  # Grid extent in latitude and longitude (degrees)

# Dummy data (replace with real-world data)
recovery_positions = np.random.uniform(low=(0, 0), high=(10, 10), size=(20, 2))  # Random recovery positions
recovery_times = np.random.randint(low=1, high=30, size=20)  # Random recovery times (days)

# Generate synthetic ocean current and wind data (replace with actual data loading)
ocean_current_speeds = np.random.uniform(low=0.1, high=1.0, size=grid_resolution)
ocean_current_directions = np.random.uniform(low=0, high=360, size=grid_resolution)
wind_speeds = np.random.uniform(low=1.0, high=10.0, size=grid_resolution)
wind_directions = np.random.uniform(low=0, high=360, size=grid_resolution)

# Leeway modeling function (dummy implementation)
def calculate_leeway_velocity(wind_speed, wind_direction):
    leeway_percentage = 0.02  # Leeway as percentage of wind speed
    leeway_speed = leeway_percentage * wind_speed
    leeway_direction = wind_direction  # Simplified assumption (leeway aligned with wind direction)
    leeway_velocity = leeway_speed * np.array([np.cos(np.radians(leeway_direction)),
                                               np.sin(np.radians(leeway_direction))])
    return leeway_velocity

# Function to simulate reverse drift trajectory
def simulate_reverse_drift(recovery_position, recovery_time, grid_extent,
                           ocean_current_speeds, ocean_current_directions,
                           wind_speeds, wind_directions, grid_resolution):
    # Initialize trajectory
    trajectory = [recovery_position]
    current_position = np.array(recovery_position)
    
    # Time step in days
    time_step = 1  # Assuming 1 day time step
    
    # Simulate reverse drift over time
    for t in range(recovery_time, 0, -time_step):
        # Interpolate ocean current and wind data at current position
        current_lat_index = int(np.interp(current_position[0], grid_extent[0], [0, grid_resolution[0]-1]))
        current_lon_index = int(np.interp(current_position[1], grid_extent[1], [0, grid_resolution[1]-1]))
        
        current_speed = ocean_current_speeds[current_lat_index, current_lon_index]
        current_direction = ocean_current_directions[current_lat_index, current_lon_index]
        wind_speed = wind_speeds[current_lat_index, current_lon_index]
        wind_direction = wind_directions[current_lat_index, current_lon_index]
        
        # Calculate leeway velocity based on wind data
        leeway_velocity = calculate_leeway_velocity(wind_speed, wind_direction)
        
        # Calculate total velocity (current + leeway)
        current_velocity = current_speed * np.array([np.cos(np.radians(current_direction)),
                                                     np.sin(np.radians(current_direction))])
        total_velocity = current_velocity + leeway_velocity
        
        # Update position based on total velocity and time step
        current_position += total_velocity * time_step
        
        # Append new position to trajectory
        trajectory.append(current_position.copy())
    
    return np.array(trajectory)

# Simulate reverse drift trajectories for each recovery event
rd_trajectories = []
for i in range(len(recovery_positions)):
    recovery_position = recovery_positions[i]
    recovery_time = recovery_times[i]
    
    # Simulate reverse drift trajectory
    rd_trajectory = simulate_reverse_drift(recovery_position, recovery_time, grid_extent,
                                           ocean_current_speeds, ocean_current_directions,
                                           wind_speeds, wind_directions, grid_resolution)
    
    rd_trajectories.append(rd_trajectory)

# Flatten all trajectories into a single array of positions
all_positions = np.concatenate(rd_trajectories, axis=0)

# Compute KDE for RD distribution
kde = KernelDensity(bandwidth=0.1)  # Bandwidth parameter (adjust for optimal smoothing)
kde.fit(all_positions)  # Fit KDE model to all positions

# Generate grid points for RD distribution visualization
x_grid, y_grid = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

# Evaluate KDE model on grid points to get RD distribution
log_density = kde.score_samples(grid_points)
rd_distribution = np.exp(log_density).reshape(x_grid.shape)

# Constants for Earth's radius and conversion
EARTH_RADIUS_NM = 3440.065  # Radius of the Earth in nautical miles (NM)

# Last Known Position (LKP) of the plane (latitude and longitude in degrees)
lkp_latitude = 40.0
lkp_longitude = -70.0

# Radius of the circular area in nautical miles (NM)
radius_nm = 40.0

# Convert radius from NM to degrees (latitude and longitude)
latitude_degrees = radius_nm / 60.0  # 1 degree of latitude is approximately 60 NM
longitude_degrees = radius_nm / (EARTH_RADIUS_NM * np.cos(np.radians(lkp_latitude)))

# Calculate latitude and longitude bounds for circular area
min_latitude = lkp_latitude - latitude_degrees
max_latitude = lkp_latitude + latitude_degrees
min_longitude = lkp_longitude - longitude_degrees
max_longitude = lkp_longitude + longitude_degrees

# Create a boolean mask to identify grid points within the circular area
mask = (x_grid >= min_longitude) & (x_grid <= max_longitude) & (y_grid >= min_latitude) & (y_grid <= max_latitude)

# Apply the mask to truncate the RD distribution outside the circular area
truncated_rd_distribution = np.where(mask, rd_distribution, 0.0)

# Visualize truncated RD distribution within circular area
plt.figure(figsize=(10, 10))
plt.contourf(x_grid, y_grid, truncated_rd_distribution, cmap='viridis', levels=50)
plt.colorbar(label='Probability Density')
plt.scatter(recovery_positions[:, 0], recovery_positions[:, 1], color='red', label='Recovery Position', zorder=10)
plt.scatter(lkp_longitude, lkp_latitude, color='blue', label='Last Known Position (LKP)', zorder=10, marker='x', s=100)
plt.title(f'Truncated RD Distribution within {radius_nm} NM Circle (LKP at {lkp_longitude}, {lkp_latitude})')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.show()
