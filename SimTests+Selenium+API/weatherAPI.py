import requests
import numpy as np

API_key = open('api_key.txt','r').read()
lat = 50
lon = 50

for i in range(50):
    lat = np.random.randint(-89,89)
    lon = np.random.randint(-179,179)
    api_call = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_key}"
    response = requests.get(api_call).json()
    if response['cod']==200:
        print(i)
    else:
        break

