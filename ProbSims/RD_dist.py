import numpy as np
import matplotlib.pyplot as plt

# Define parameters and data (dummy dataset)
num_bodies = 10
recovery_positions = np.random.rand(num_bodies, 2)  # Random recovery positions (latitude, longitude)
recovery_times = np.random.randint(1, 100, size=num_bodies)  # Random recovery times (in days)
grid_size = (10, 10)  # Size of the grid for ocean current and wind data
current_speeds = np.random.uniform(0.1, 1.0, size=grid_size)  # Random current speeds (m/s)
current_directions = np.random.uniform(0, 360, size=grid_size)  # Random current directions (degrees)
wind_speeds = np.random.uniform(1.0, 10.0, size=grid_size)  # Random wind speeds (m/s)
wind_directions = np.random.uniform(0, 360, size=grid_size)  # Random wind directions (degrees)

# Leeway model parameters (dummy values)
leeway_percentage = 0.02  # Leeway as percentage of wind speed
time_step_minutes = 60  # Time step in minutes for simulation
simulation_duration_days = 30  # Duration of reverse drift simulation in days

# Function to simulate reverse drift trajectory
def simulate_reverse_drift(recovery_position, recovery_time):
    position_history = [recovery_position]
    current_position = np.array(recovery_position)
    
    # Simulate reverse drift over time
    for t in range(recovery_time * 24 * 60, 0, -time_step_minutes):  # Convert recovery_time to minutes
        # Calculate current time in days (negative value, moving backwards in time)
        current_time_days = -t / (24 * 60)
        
        # Determine current grid cell based on current position
        grid_cell = (int(current_position[0] * grid_size[0]), int(current_position[1] * grid_size[1]))
        
        # Get current speed and direction from grid data
        current_speed = current_speeds[grid_cell]
        current_direction = np.radians(current_directions[grid_cell])
        
        # Calculate current velocity components (east-west and north-south)
        current_velocity = current_speed * np.array([np.cos(current_direction), np.sin(current_direction)])
        
        # Calculate leeway velocity based on wind speed and leeway percentage
        wind_speed = wind_speeds[grid_cell]
        leeway_velocity = leeway_percentage * wind_speed
        
        # Combine ocean current velocity and leeway velocity
        total_velocity = current_velocity + leeway_velocity
        
        # Update current position based on total velocity and time step
        current_position += total_velocity * (time_step_minutes / (60 * 24))  # Convert time_step_minutes to days
        
        # Append updated position to history
        position_history.append(current_position.copy())
        
        # Break loop if simulation duration is reached
        if current_time_days <= -simulation_duration_days:
            break
    
    return np.array(position_history)

# Simulate reverse drift for each recovery body
rd_trajectories = []
for i in range(num_bodies):
    recovery_position = recovery_positions[i]
    recovery_time = recovery_times[i]
    rd_trajectory = simulate_reverse_drift(recovery_position, recovery_time)
    rd_trajectories.append(rd_trajectory)

# Plot reverse drift trajectories
plt.figure(figsize=(8, 8))
for trajectory in rd_trajectories:
    plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.5)
plt.scatter(recovery_positions[:, 0], recovery_positions[:, 1], color='red', label='Recovery Position')
plt.title('Reverse Drift Trajectories')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.legend()
plt.grid(True)
plt.show()
