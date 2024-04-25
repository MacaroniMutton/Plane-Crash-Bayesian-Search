import requests
import json

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

# Example usage:
latitude = 2.9971
longitude = 84.0938
start_time = "2024-04-23T07:00:00Z"
end_time = "2024-04-24T07:00:00Z"
interval = "PT1H"

# Query Meteomatics API
result = query_meteomatics_api(latitude, longitude, start_time, end_time, interval)

# Print the result
if result is not None:
    print(json.dumps(result, indent=2))
else:
    print("API query failed.")
