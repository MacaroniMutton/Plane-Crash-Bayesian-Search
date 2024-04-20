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


def normalize_longitude(lon):
    while lon < -180:
        lon += 360
    while lon > 180:
        lon -= 360
    return lon

def download_bathymetry_data(lat, lng):
    download_path = "C:\\Users\\Manan Kher\\OneDrive\\Documents\\Plane-Crash-Bayesian-Search\\Djano_Pygbag_Website\\mysite\\myapp"

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
    north_p = round(lat + 1.5, 4)
    south_p = round(lat - 1.5, 4)
    east_p = round(lng + 1.5, 4)
    west_p = round(lng - 1.5, 4)

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


def load_bathymetry_data(dataset_path):
    CELL_SIZE = 8
    ROWS = 96
    COLUMNS = 96

    data = xr.load_dataset(dataset_path)
    elevation = data.elevation
    li = np.array([[0]*COLUMNS for _ in range(ROWS)])
    mini = float('inf')
    maxi = float('-inf')
    for lat in range(96):
        for lon in range(96):
            li[95-lat][lon] = (elevation[(lat)*int(data.sizes['lat']/96):(lat+1)*int(data.sizes['lat']/96), (lon)*int(data.sizes['lon']/96):(lon+1)*int(data.sizes['lon']/96)].mean().load())
            mini = min(mini, li[95-lat][lon])
            maxi = max(maxi, li[95-lat][lon])
    li = (li-mini)
    li = li/(maxi-mini)
    print(li)
    with open("C:\\Users\\Manan Kher\\OneDrive\\Documents\\Plane-Crash-Bayesian-Search\\Djano_Pygbag_Website\\bathymetry_data.txt", "wb") as fp:   #Pickling
        pickle.dump(li, fp)
    return li



