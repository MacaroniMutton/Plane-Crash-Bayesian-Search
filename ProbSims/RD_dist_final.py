import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# Define grid parameters
grid_resolution = (96, 96)  # Grid resolution (number of cells)
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
        current_position -= total_velocity * time_step
        
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

# Visualize reverse drift trajectories
# plt.figure(figsize=(10, 10))
# for trajectory in rd_trajectories:
#     plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.5)
# plt.scatter(recovery_positions[:, 0], recovery_positions[:, 1], color='red', label='Recovery Position')
# plt.title('Reverse Drift Trajectories')
# plt.xlabel('Latitude')
# plt.ylabel('Longitude')
# plt.legend()
# plt.grid(True)
# plt.show()

# print(rd_trajectories)

# Generate RD distribution (probability density estimation)
# Example: Calculate kernel density estimation (KDE) for RD distribution
from sklearn.neighbors import KernelDensity
import seaborn as sns

# Flatten all trajectories into a single array of positions
all_positions = np.concatenate(rd_trajectories, axis=0)

# Compute KDE for RD distribution
kde = KernelDensity(bandwidth=0.1)  # Bandwidth parameter (adjust for optimal smoothing)
kde.fit(all_positions)  # Fit KDE model to all positions

# Generate grid points for RD distribution visualization
x_grid, y_grid = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
# print(x_grid, y_grid)
# print(grid_points)


# Evaluate KDE model on grid points to get RD distribution
log_density = kde.score_samples(grid_points)
rd_distribution = np.exp(log_density).reshape(x_grid.shape)
# print(rd_distribution.shape)
# sns.heatmap(rd_distribution)

# Visualize RD distribution
# plt.figure(figsize=(10, 10))
# plt.contourf(x_grid, y_grid, rd_distribution, cmap='viridis', levels=50)
# plt.colorbar(label='Probability Density')
# plt.scatter(recovery_positions[:, 0], recovery_positions[:, 1], color='red', label='Recovery Position', zorder=10)
# plt.title('Reverse Drift Probability Density (RD Distribution)')
# plt.xlabel('Latitude')
# plt.ylabel('Longitude')
# plt.legend()
# plt.grid(True)
# plt.show()



# Plot both trajectories and RD distribution side by side
fig, axs = plt.subplots(1, 2, figsize=(16, 8))

# Plot reverse drift trajectories
for trajectory in rd_trajectories:
    axs[0].plot(trajectory[:, 0], trajectory[:, 1], alpha=0.5)
axs[0].scatter(recovery_positions[:, 0], recovery_positions[:, 1], color='red', label='Recovery Position')
axs[0].set_title('Reverse Drift Trajectories')
axs[0].set_xlabel('Latitude')
axs[0].set_ylabel('Longitude')
axs[0].legend()
axs[0].grid(True)

# Plot RD distribution
contour = axs[1].contourf(x_grid, y_grid, rd_distribution, cmap='viridis', levels=50)
plt.colorbar(contour, ax=axs[1], label='Probability Density')
axs[1].scatter(recovery_positions[:, 0], recovery_positions[:, 1], color='red', label='Recovery Position')
axs[1].set_title('Reverse Drift Probability Density (RD Distribution)')
axs[1].set_xlabel('Latitude')
axs[1].set_ylabel('Longitude')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()  # Adjust subplot layout to prevent overlap
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # Define grid parameters for the larger 10-degree range
# grid_resolution_large = (100, 100)  # Grid resolution for the larger 10-degree range
# grid_extent_large = ((0, 10), (0, 10))  # Grid extent for the larger range in latitude and longitude (degrees)

# # Define grid parameters for the smaller 1.5-degree range within the 10-degree range
# subgrid_resolution = (96, 96)  # Grid resolution for the 1.5-degree range (96x96 grid)
# subgrid_extent = ((0, 1.5), (0, 1.5))  # Grid extent for the 1.5-degree range in latitude and longitude (degrees)

# # Generate example RD distribution (for illustration) over the larger grid
# rd_distribution_large = np.random.uniform(low=0, high=1, size=grid_resolution_large)

# # Determine the indices corresponding to the cells of the 96x96 grid within the larger 10-degree grid
# min_lat_index = int(np.interp(subgrid_extent[0][0], grid_extent_large[0], [0, grid_resolution_large[0]-1]))
# max_lat_index = int(np.interp(subgrid_extent[0][1], grid_extent_large[0], [0, grid_resolution_large[0]-1]))

# min_lon_index = int(np.interp(subgrid_extent[1][0], grid_extent_large[1], [0, grid_resolution_large[1]-1]))
# max_lon_index = int(np.interp(subgrid_extent[1][1], grid_extent_large[1], [0, grid_resolution_large[1]-1]))

# # Extract the subset of the RD distribution for the 96x96 subgrid within the larger grid
# rd_distribution_subgrid = rd_distribution[min_lat_index:max_lat_index+1, min_lon_index:max_lon_index+1]

# # Create grid of latitude and longitude values for the subgrid
# lat_values = np.linspace(subgrid_extent[0][0], subgrid_extent[0][1], subgrid_resolution[0])
# lon_values = np.linspace(subgrid_extent[1][0], subgrid_extent[1][1], subgrid_resolution[1])

# # Plotting the extracted subset of RD distribution for the 96x96 subgrid
# plt.figure(figsize=(8, 8))
# plt.imshow(rd_distribution_subgrid, origin='lower', extent=(lon_values[0], lon_values[-1], lat_values[0], lat_values[-1]), cmap='viridis')
# plt.colorbar(label='Probability Density')
# plt.title('Truncated RD Distribution (1.5 degrees x 1.5 degrees, 96x96 subgrid)')
# plt.xlabel('Longitude (degrees)')
# plt.ylabel('Latitude (degrees)')
# plt.grid(True)
# plt.show()







# import numpy as np
# import matplotlib.pyplot as plt

# # Define function to truncate RD distribution within a circular region (40 NM radius)
# def truncate_rd_distribution(rd_distribution, lkp_latitude, lkp_longitude, grid_resolution, grid_extent):
#     # Convert 40 NM radius to degrees of latitude and longitude
#     nm_to_deg = 1 / 60.0  # 1 degree = 60 NM
#     radius_nm = 40.0  # Radius in NM
#     radius_deg_lat = (radius_nm * nm_to_deg) / 2  # Radius in degrees of latitude
#     radius_deg_lon = (radius_nm * nm_to_deg) / (2 * np.cos(np.radians(lkp_latitude)))  # Radius in degrees of longitude

#     # Create grid of latitude and longitude values
#     lat_grid = np.linspace(grid_extent[0][0], grid_extent[0][1], grid_resolution[0])
#     lon_grid = np.linspace(grid_extent[1][0], grid_extent[1][1], grid_resolution[1])

#     # Create meshgrid of latitude and longitude
#     lon_grid_2d, lat_grid_2d = np.meshgrid(lon_grid, lat_grid)

#     # Calculate distances from LKP
#     dist_lat = lat_grid_2d - lkp_latitude
#     dist_lon = lon_grid_2d - lkp_longitude

#     # Create circular mask within 40 NM radius
#     mask = (dist_lat**2 / radius_deg_lat**2 + dist_lon**2 / radius_deg_lon**2) <= 1.0

#     # Apply mask to RD distribution
#     truncated_rd_distribution = rd_distribution.copy()
#     truncated_rd_distribution[~mask] = 0.0  # Set values outside circle to zero

#     return truncated_rd_distribution

# # Example usage:
# # Assuming rd_distribution is the computed RD distribution 2D array

# # Last Known Position (LKP) of the plane (latitude, longitude)
# lkp_latitude = 30.0  # Example latitude of LKP
# lkp_longitude = -120.0  # Example longitude of LKP

# # Grid extent (latitude, longitude) and resolution (number of cells)
# # grid_extent = ((0, 60), (-180, 180))  # Example grid extent (degrees)
# # grid_resolution = (100, 100)  # Example grid resolution

# # Truncate RD distribution within 40 NM circle centered at LKP
# truncated_rd_distribution = truncate_rd_distribution(rd_distribution, lkp_latitude, lkp_longitude, grid_resolution, grid_extent)

# # Visualize truncated RD distribution
# plt.figure(figsize=(10, 10))
# plt.contourf(lon_grid_2d, lat_grid_2d, truncated_rd_distribution, cmap='viridis', levels=50)
# plt.colorbar(label='Probability Density')
# plt.scatter(lkp_longitude, lkp_latitude, color='red', label='LKP', zorder=10)
# plt.title('Truncated Reverse Drift Probability Density (40 NM Radius)')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.legend()
# plt.grid(True)
# plt.show()

