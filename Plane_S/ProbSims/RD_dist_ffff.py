import numpy as np
import matplotlib.pyplot as plt
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

# Flatten all trajectories into a single array of positions
all_positions = np.concatenate(rd_trajectories, axis=0)

# Calculate endpoint positions of each trajectory
end_points = np.array([trajectory[-1] for trajectory in rd_trajectories])

# Calculate distances from all positions to the endpoints
distances = np.linalg.norm(all_positions[:, np.newaxis] - end_points, axis=2)
max_distances = np.max(distances, axis=1)

# Use distances as weights for KDE (higher weight for positions closer to endpoints)
weights = 1 - (distances / max_distances[:, np.newaxis])

# Reshape weights to match the expected shape for sample weights
weights_flat = weights.ravel()[:len(all_positions)]

# Compute KDE for RD distribution with weighted samples
kde = KernelDensity(bandwidth=0.1)  # Bandwidth parameter (adjust for optimal smoothing)
kde.fit(all_positions, sample_weight=weights_flat)  # Fit KDE model with weighted samples

# Generate grid points for RD distribution visualization
x_grid, y_grid = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

# Evaluate KDE model on grid points to get RD distribution
log_density = kde.score_samples(grid_points)
rd_distribution = np.exp(log_density).reshape(x_grid.shape)

# Plotting both trajectories and RD distribution side by side
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
