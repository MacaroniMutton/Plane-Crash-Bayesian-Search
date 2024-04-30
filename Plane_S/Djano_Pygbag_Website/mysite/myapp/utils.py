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


def city_lat_lng(city):
    api_url = 'https://api.api-ninjas.com/v1/geocoding?city={}'.format(city)
    with open('myapp\city_api_key.txt','r') as f:
        api_key = f.read()
    response = requests.get(api_url, headers={'X-Api-Key': api_key})
    if response.status_code == requests.codes.ok:
        response = json.loads(response.text)
        print(response[0])
        lat = response[0]["latitude"]
        lng = response[0]["longitude"]
        return lat, lng
    else:
        return f"Error: {response.status_code, response.text}"
    
def distance(a, b):
    x1, y1 = a
    x2, y2 = b
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points given their latitude and longitude coordinates.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Radius of the Earth in kilometers (approximate)
    radius_earth_km = 6371
    
    # Calculate the distance
    distance_km = radius_earth_km * c
    
    return distance_km
    
def calculate_vector_angle(point1, point2):
    # Extract latitude and longitude from the points
    lat1, lon1 = point1["lat"], point1["lng"]
    lat2, lon2 = point2["lat"], point2["lng"]

    # Calculate the differences in longitude and latitude
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1

    # Calculate the angle using arctangent function (atan2)
    angle_rad = math.atan2(delta_lat, delta_lon)

    # Convert angle from radians to degrees
    angle_deg = math.degrees(angle_rad)

    # Normalize angle to be within [0, 360) degrees
    normalized_angle = (angle_deg + 360) % 360

    return normalized_angle

def makeGaussian(size, a, b, theta, center=None):

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    theta = (theta)*math.pi/180

    # return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    return np.exp(-4*np.log(2) * (((x-x0)*math.cos(theta)-(y-y0)*math.sin(theta))**2 / a**2 + ((y-y0)*math.cos(theta)+(x-x0)*math.sin(theta))**2 / b**2))

def makeCircularUniform(size, radius, center=None):
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    # Calculate distances from the center
    distances = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    # Create circular mask
    circular_mask = (distances <= radius).astype(float)

    return circular_mask

def normalize_longitude(lon):
    while lon < -180:
        lon += 360
    while lon > 180:
        lon -= 360
    return lon

def download_bathymetry_data(lat, lng):
    download_path = "C:\\Users\\Manan Kher\\OneDrive\\Documents\\MINI_PROJECT\\Plane-Crash-Bayesian-Search\\Plane_S\\Djano_Pygbag_Website\\mysite\\myapp"

    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_experimental_option("prefs", {
        "download.default_directory": download_path,
        "download.prompt_for_download": False,  # Optional, to suppress download prompt
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    # service = Service(executable_path='<path-to-chrome>')
    web = webdriver.Chrome(options=options)
    web.get('https://download.gebco.net/')
    time.sleep(2)

    lng = normalize_longitude(lng)
    print(lat, lng)
    print(type(lat), type(lng))
    north_p = round(lat + 0.75, 4)
    south_p = round(lat - 0.75, 4)
    east_p = round(lng + 0.75, 4)
    west_p = round(lng - 0.75, 4)

    north = web.find_element('xpath','//*[@id="coordinates-card"]/div[2]/div[1]/div[1]/div/input')
    north.clear()
    north.send_keys(Keys.CONTROL + "a")
    north.send_keys(f"{north_p}")


    south = web.find_element('xpath','//*[@id="coordinates-card"]/div[2]/div[3]/div[2]/div/input')
    south.clear()
    south.send_keys(Keys.CONTROL + "a")
    south.send_keys(f"{south_p}")


    east = web.find_element('xpath','//*[@id="coordinates-card"]/div[2]/div[2]/div[3]/div/input')
    east.clear()
    east.send_keys(Keys.CONTROL + "a")
    east.send_keys(f"{east_p}")


    west = web.find_element('xpath','//*[@id="coordinates-card"]/div[2]/div[2]/div[1]/div/input')
    west.clear()
    west.send_keys(Keys.CONTROL + "a")
    west.send_keys(f"{west_p}")


    inp = web.find_element('xpath','//*[@id="gridCheck"]')
    inp.click()
    time.sleep(1)

    shaded_relief = web.find_element('xpath','/html/body/div[1]/div[2]/div/form/div[3]/div[2]/table/tbody/tr[7]/td[3]/div/div/input')
    shaded_relief.click()
    time.sleep(1)

    add_basket = web.find_element('xpath','//*[@id="sidebar-add-to-basket"]')
    add_basket.click()
    time.sleep(2)

    view_basket = web.find_element('xpath','//*[@id="data-selection-card"]/div[3]/button[2]')
    view_basket.click()
    time.sleep(2)

    download = web.find_element('xpath','//*[@id="basket-download-button"]')
    download.click()


    files = os.listdir(download_path)
    gebco_file_exists = any(file.startswith("GEBCO") for file in files)


    while not gebco_file_exists:
        time.sleep(1)
        files = os.listdir(download_path)
        gebco_file_exists = any(file.startswith("GEBCO") for file in files)


def extract_zip(zip_file_path, extract_to_directory):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_directory)


def find_gebco_nc_file(directory):
    # Iterate over the files in the directory
    for filename in os.listdir(directory):
        # Check if the filename starts with "gebco" and ends with ".nc"
        if filename.startswith("gebco") and filename.lower().endswith(".nc"):
            return os.path.join(directory, filename)  # Return the full path to the matching file
    return None  # Return None if no matching file is found

def find_shaded_relief(directory):
    # Iterate over the files in the directory
    for filename in os.listdir(directory):
        # Check if the filename starts with "gebco" and ends with ".nc"
        if filename.startswith("gebco") and filename.lower().endswith(".png"):
            return os.path.join(directory, filename)  # Return the full path to the matching file
    return None  # Return None if no matching file is found


def load_bathymetry_data(dataset_path):
    CELL_SIZE = 8
    ROWS = 96
    COLUMNS = 96

    data = xr.load_dataset(dataset_path)
    elevation = data.elevation
    li = np.array([[0]*COLUMNS for _ in range(ROWS)])
    lat_lng_li = [[None]*COLUMNS for _ in range(ROWS)]
    mini = float('inf')
    maxi = float('-inf')
    for lat in range(96):
        for lon in range(96):
            lat_lng_li[95-lat][lon] = [[data.lat[int(lat*(data.sizes['lat']/96))], data.lat[int((lat+1)*(data.sizes['lat']/96))-1]], [data.lon[int(lon*(data.sizes['lon']/96))], data.lon[int((lon+1)*(data.sizes['lon']/96))-1]]]
            li[95-lat][lon] = (elevation[int(lat*(data.sizes['lat']/96)):int((lat+1)*(data.sizes['lat']/96))-1, int(lon*(data.sizes['lon']/96)):int((lon+1)*(data.sizes['lon']/96))-1].mean().load())
            mini = min(mini, li[95-lat][lon])
            maxi = max(maxi, li[95-lat][lon])
    lat_lng_li = np.array(lat_lng_li)
    li = (li-mini)
    li = li/(maxi-mini)
    print(li)
    with open("C:\\Users\\Manan Kher\\OneDrive\\Documents\\MINI_PROJECT\\Plane-Crash-Bayesian-Search\\Plane_S\\Djano_Pygbag_Website\\bathymetry_data.txt", "wb") as fp:   #Pickling
        pickle.dump(li, fp)
    return li, lat_lng_li


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
    username = "spit_khadhav_manushka"
    password = "r8W5D5yTbG"

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

def find_cell(lat, lon, lat_lng_li):
    # for i in range(len(lat_lng_li)):
    #     for j in range(len(lat_lng_li[0])):
    #         lat_range = lat_lng_li[i][j][0]
    #         lon_range = lat_lng_li[i][j][1]
    #         if lat_range[0] <= lat <= lat_range[1] and lon_range[0] <= lon <= lon_range[1]:
    #             return [i, j]
    # return None
    point = [lat, lon]
    lat_li = []
    lng_li = []

    for i in range(len(lat_lng_li)):
        lat_li.append((lat_lng_li[i][0][0][0]+lat_lng_li[i][0][0][1])/2)

    print(len(lat_li))

    for i in range(len(lat_lng_li[0])):
        lng_li.append((lat_lng_li[0][i][1][0]+lat_lng_li[0][i][1][1])/2)

    print(lng_li)

    lat_distances = [abs(lat-point[0]) for lat in lat_li]
    lng_distances = [abs(lng-point[1]) for lng in lng_li]
    row = lat_distances.index(min(lat_distances))
    col = lng_distances.index(min(lng_distances))
    return [row, col]


def plot_reverse_drift_trajectories(recovered_bodies, lat_lng_li, lkp_lat, lkp_lng):
    print(lat_lng_li)
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

    recovery_endpoints = [trajectory[-1] for trajectory in rd_trajectories]
    recovery_endpoints = np.array(recovery_endpoints)
    endpoint_weight=10.0

    # Compute KDE for RD distribution with variable bandwidth
    distances = np.array(distances)
    max_distance = np.max(distances)
    bandwidths = 0.02 + 0.2 * (distances / max_distance)  # Dynamic bandwidth based on distance

    kde = KernelDensity(bandwidth=bandwidths.mean())  # Use mean bandwidth initially
    
    # Weighting positions based on proximity to recovery endpoints
    weights = np.ones(len(all_positions))
    for endpoint in recovery_endpoints:
        distances = np.linalg.norm(all_positions - endpoint, axis=1)
        weights += endpoint_weight * np.exp(-distances**2 / (2 * bandwidths.mean()**2))

    kde.fit(all_positions, sample_weight=weights)  # Fit KDE model to all positions

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

    # Save the figure
    plt.savefig('C:\\Users\\Manan Kher\\OneDrive\\Documents\\MINI_PROJECT\\Plane-Crash-Bayesian-Search\\Plane_S\\Djano_Pygbag_Website\\mysite\\myapp\\static\\myapp\\images\\my_plot.png')

    bottom_left = [np.min(all_positions[:, 0]), np.min(all_positions[:, 1])]
    top_left = [np.max(all_positions[:, 0]), np.min(all_positions[:, 1])]
    top_right = [np.max(all_positions[:, 0]), np.max(all_positions[:, 1])]
    bottom_right = [np.min(all_positions[:, 0]), np.max(all_positions[:, 1])]
    
    print(top_right, top_left, bottom_left, bottom_right)

    bottom_left = find_cell(bottom_left[0], bottom_left[1], lat_lng_li)
    top_left = find_cell(top_left[0], top_left[1], lat_lng_li)
    top_right = find_cell(top_right[0], top_right[1], lat_lng_li)
    bottom_right = find_cell(bottom_right[0], bottom_right[1], lat_lng_li)

    print(top_right, top_left, bottom_left, bottom_right)


    lon_grid, lat_grid = np.meshgrid(np.linspace(lkp_lng-0.75, lkp_lng+0.75, num=96),
                                     np.linspace(lkp_lat-0.75, lkp_lat+0.75, num=96))
    points_grid = np.vstack([lat_grid.ravel(), lon_grid.ravel()]).T

    # Evaluate KDE model on grid points
    log_density_grid = kde.score_samples(points_grid)
    shrinked_grid = np.exp(log_density_grid).reshape(lon_grid.shape)
    mini = shrinked_grid.min()
    maxi = shrinked_grid.max()
    shrinked_grid = shrinked_grid-mini
    shrinked_grid = shrinked_grid/(maxi-mini)
    
    # shrinked_rd_grid = np.zeros((96,96))
    # shrinked_rd_grid[top_right[0]:bottom_right[0], top_left[1]:top_right[1]] = shrinked_grid


    # Return the density grid as a 96x96 array
    return density_grid[::-1], shrinked_grid[::-1]

