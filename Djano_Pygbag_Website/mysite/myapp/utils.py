import requests
import json
import math
import numpy as np

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
