
import time
from datetime import datetime
import requests
import numpy as np

date_time_str = '2024-03-21 00:00:00' # Example date and time
date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
unix_time = int(time.mktime(date_time_obj.timetuple()))
start = unix_time

date_time_str = '2024-04-21 00:00:00' # Example date and time
date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
unix_time = int(time.mktime(date_time_obj.timetuple()))
end = unix_time

API_key = open('api_key.txt','r').read()
lat = 50
lon = 50


# lat = np.random.randint(-89,89)
# lon = np.random.randint(-179,179)
# api_call = f"https://history.openweathermap.org/data/2.5/history/city?lat={lat}&lon={lon}&type=hour&start={start}&end={end}&appid={API_key}"
api_call = f"https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&hourly=temperature_2m,wind_speed_10m,wind_direction_10m&past_days=1"
response = requests.get(api_call).json()
print(response['hourly']['time'][:24])
print(response['hourly']['wind_speed_10m'][:24])
print(response['hourly']['wind_direction_10m'][:24])
    # if response['cod']==200:
    #     print(response)
    # else:
    #     print()

