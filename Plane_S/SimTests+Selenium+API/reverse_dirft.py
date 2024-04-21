from datetime import datetime, timedelta
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import requests
import json

# Define a single crash time
crash_time = datetime(2024, 4, 20, 1, 0, 0) 

# Define lists of recovery positions and recovery times
recovery_positions = np.array([
    [-24, -20],
    [-23.85, -18.99],
    # [-23.2, -19.5],
])

recovery_times = [
    datetime(2024, 4, 21, 23, 0, 0), 
    datetime(2024, 4, 20, 8, 0, 0), 
    # datetime(2024, 4, 21, 1, 0, 0),   
]



def query_meteomatics_api(latitude, longitude, start_time, end_time, interval='PT1H'):
    """
    Query Meteomatics API for ocean current direction, speed, and wind data at a specific location and time interval.

    Parameters:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        start_time (str): Start time of the query in ISO 8601 format.
        end_time (str): End time of the query in ISO 8601 format.
        interval (str, optional): Time interval for the query (default is 'PT1H').

    Returns:
        dict: JSON response from the Meteomatics API.
    """
    # Define your Meteomatics API credentials
    username = "spit_mutton_macaroni"
    password = "TQ4mj36VFa"

    # Define additional parameters for the API query
    parameters = "ocean_current_direction:d,ocean_current_speed_2m:kmh,wind_dir_FL10:d,wind_speed_FL10:kmh"

    # Construct the API URL based on input parameters
    api_url = f"https://api.meteomatics.com/{start_time}--{end_time}:{interval}/{parameters}/{latitude},{longitude}/json?model=mix"

    try:
        # Make a GET request with basic authentication
        response = requests.get(api_url, auth=(username, password))

        # Check if the request was successful (HTTP status code 200)
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            return data
        else:
            print(f"Request failed with status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return None
    

# Leeway modeling function (dummy implementation)
def calculate_leeway_velocity(wind_speed, wind_direction):
    leeway_percentage = 0.02  # Leeway as percentage of wind speed
    leeway_speed = leeway_percentage * wind_speed
    leeway_direction = wind_direction  # Simplified assumption (leeway aligned with wind direction)
    leeway_velocity = leeway_speed * np.array([np.cos(np.radians(leeway_direction)),
                                               np.sin(np.radians(leeway_direction))])
    return leeway_velocity



def simulate_reverse_drift(recovery_position, crash_time, recovery_time):

    recovery_time_hours = (recovery_time - crash_time).total_seconds() / 3600
    recovery_time_hours = int(recovery_time_hours)  
    latitude, longitude = recovery_position
    # Initialize trajectory
    trajectory = [recovery_position]
    current_position = np.array(recovery_position)
    
    # Time step in hours
    time_step = 1  # Assuming 1 hour time step

    start_time = datetime(crash_time.year, crash_time.month, crash_time.day, 0, 0, 0)
    start_time = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_time = datetime(recovery_time.year, recovery_time.month, recovery_time.day+1, 0, 0, 0)
    end_time = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    recov_time_check = recovery_time

    # Simulate reverse drift over time
    while recovery_time_hours>0:
        print(latitude, longitude)
        api_data = query_meteomatics_api(round(latitude,3), round(longitude,3), start_time, end_time)

        
        ocean_current_direction = None
        ocean_current_speed = None
        wind_direction = None
        wind_speed = None

        for obj in api_data["data"][0]["coordinates"][0]["dates"]:
            date = obj["date"]
            hour_date = date.split(":")[0]
            if hour_date == recov_time_check.strftime("%Y-%m-%dT%H:%M:%SZ").split(":")[0]:
                ocean_current_direction = obj["value"]
                break

        for obj in api_data["data"][1]["coordinates"][0]["dates"]:
            date = obj["date"]
            hour_date = date.split(":")[0]
            if hour_date == recov_time_check.strftime("%Y-%m-%dT%H:%M:%SZ").split(":")[0]:
                ocean_current_speed = obj["value"]
                print(ocean_current_speed)
                break

        for obj in api_data["data"][2]["coordinates"][0]["dates"]:
            date = obj["date"]
            hour_date = date.split(":")[0]
            if hour_date == recov_time_check.strftime("%Y-%m-%dT%H:%M:%SZ").split(":")[0]:
                wind_direction = obj["value"]
                break

        for obj in api_data["data"][3]["coordinates"][0]["dates"]:
            date = obj["date"]
            hour_date = date.split(":")[0]
            if hour_date == recov_time_check.strftime("%Y-%m-%dT%H:%M:%SZ").split(":")[0]:
                wind_speed = obj["value"]
                print(wind_speed)
                break

        recov_time_check = recov_time_check - timedelta(hours=1)
            
        recovery_time_hours -= 1

        # Calculate leeway velocity based on wind data
        leeway_velocity = calculate_leeway_velocity(wind_speed, wind_direction)
        
        # Calculate total velocity (current + leeway)
        current_velocity = ocean_current_speed * np.array([np.cos(np.radians(ocean_current_direction)),
                                                     np.sin(np.radians(ocean_current_direction))])
        total_velocity = current_velocity + leeway_velocity

        print(total_velocity * time_step)

        km_per_degree = 111.32

        distance_vector = total_velocity*time_step
        degree_change_vector = distance_vector/km_per_degree
        
        # Update position based on total velocity and time step
        current_position = current_position - degree_change_vector
        print(current_position)

        latitude, longitude = current_position
        
        # Append new position to trajectory
        trajectory.append(current_position.copy())
    
    return np.array(trajectory)

# Simulate reverse drift trajectories for each recovery event
rd_trajectories = []
for i in range(len(recovery_positions)):
    recovery_position = recovery_positions[i]
    recovery_time = recovery_times[i]
    
    # Simulate reverse drift trajectory
    rd_trajectory = simulate_reverse_drift(recovery_position, crash_time, recovery_time)
    rd_trajectories.append(rd_trajectory)

# Visualize reverse drift trajectories
plt.figure(figsize=(10, 10))
for trajectory in rd_trajectories:
    plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.5)
plt.scatter(recovery_positions[:, 0], recovery_positions[:, 1], color='red', label='Recovery Position')
plt.title('Reverse Drift Trajectories')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.legend()
plt.grid(True)
plt.show()