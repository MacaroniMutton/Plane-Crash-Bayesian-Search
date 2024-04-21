import requests
import json
import math
import numpy as np
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
import time
import zipfile
import xarray as xr
import pickle
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from sklearn.neighbors import KernelDensity
import seaborn as sns




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



def simulate_reverse_drift(recovered_body):

    crash_time = recovered_body.crashTime
    recovery_time = recovered_body.recoveryTime
    recovery_position = [recovered_body.latitude, recovered_body.longitude]

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

def plot_reverse_drift_trajectories(recovered_bodies):
    rd_trajectories = []

    for recovered_body in recovered_bodies:
        rd_trajectory = simulate_reverse_drift(recovered_body)
        rd_trajectories.append(rd_trajectory)

    # Create a new figure and axes
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    # Plot reverse drift trajectories and compute distances
    all_positions = []
    distances = []
    for trajectory in rd_trajectories:
        latitudes = trajectory[:, 0]
        longitudes = trajectory[:, 1]
        ax[0].plot(longitudes, latitudes, alpha=0.5)  # Reversed latitude and longitude for plotting
        # Calculate distances along the trajectory
        dist = np.cumsum(np.sqrt(np.diff(latitudes)**2 + np.diff(longitudes)**2))
        dist = np.insert(dist, 0, 0)  # Insert 0 at the beginning to match length
        distances.extend(dist.tolist())
        all_positions.extend(list(zip(latitudes, longitudes)))

    # Plot recovery positions
    ax[0].scatter([rb.longitude for rb in recovered_bodies], 
               [rb.latitude for rb in recovered_bodies], 
               color='red', label='Recovery Position')

    # Set plot title and labels
    ax[0].set_title('Reverse Drift Trajectories')
    ax[0].set_xlabel('Longitude')
    ax[0].set_ylabel('Latitude')
    ax[0].legend()
    ax[0].grid(True)

    # Convert positions to numpy array
    all_positions = np.array(all_positions)

    # Compute KDE for RD distribution with variable bandwidth
    distances = np.array(distances)
    max_distance = np.max(distances)
    bandwidths = 0.05 + 0.2 * (distances / max_distance)  # Dynamic bandwidth based on distance

    kde = KernelDensity(bandwidth=bandwidths.mean())  # Use mean bandwidth initially
    kde.fit(all_positions)  # Fit KDE model to all positions

    # Generate grid points for visualization
    lon_grid, lat_grid = np.meshgrid(np.linspace(np.min(all_positions[:, 1]), np.max(all_positions[:, 1]), num=96),
                                     np.linspace(np.min(all_positions[:, 0]), np.max(all_positions[:, 0]), num=96))
    points_grid = np.vstack([lat_grid.ravel(), lon_grid.ravel()]).T

    # Evaluate KDE model on grid points
    log_density_grid = kde.score_samples(points_grid)
    density_grid = np.exp(log_density_grid).reshape(lon_grid.shape)

    # Plot density contours
    ax[1].contourf(lon_grid, lat_grid, density_grid, cmap='viridis')
    ax[1].scatter([rb.longitude for rb in recovered_bodies], 
               [rb.latitude for rb in recovered_bodies], 
               color='red', label='Recovery Position')
    ax[1].set_title('Reverse Drift Density Distribution')
    ax[1].set_xlabel('Longitude')
    ax[1].set_ylabel('Latitude')
    ax[1].legend()
    ax[1].grid(True)
    plt.show()

    # Save the figure
    # plt.savefig('C:\\Users\\Manan Kher\\OneDrive\\Documents\\Plane-Crash-Bayesian-Search\\Djano_Pygbag_Website\\mysite\\myapp\\static\\myapp\\images\\my_plot.png')

    # Return the density grid as a 96x96 array
    return density_grid


class RecoveredBody:
    def __init__(self, latitude, longitude, crash_time, recovery_time):
        self.latitude = latitude
        self.longitude = longitude
        self.crashTime = crash_time
        self.recoveryTime = recovery_time

    def __str__(self):
        return f"{self.latitude}, {self.longitude} - {self.recovery_time}"

# Create two instances of RecoveredBody
# Object 1
crash_time_1 = datetime(2024, 4, 20, 1, 0)  # 2024-04-20 01:00:00
recovery_time_1 = datetime(2024, 4, 20, 16, 0)  # 2024-04-20 16:00:00
recovered_body_1 = RecoveredBody(latitude=-13.3, longitude=72.78, crash_time=crash_time_1, recovery_time=recovery_time_1)

# Object 2
crash_time_2 = datetime(2024, 4, 20, 1, 0)  # 2024-04-20 01:00:00
recovery_time_2 = datetime(2024, 4, 20, 20, 0)  # 2024-04-20 20:00:00
recovered_body_2 = RecoveredBody(latitude=-13.0, longitude=73.0, crash_time=crash_time_2, recovery_time=recovery_time_2)

recoveredBodies = [recovered_body_1, recovered_body_2]

rd_dist = plot_reverse_drift_trajectories(recoveredBodies)
print(rd_dist)
sns.heatmap(rd_dist)
plt.show()
sns.heatmap(rd_dist[::-1])
plt.show()