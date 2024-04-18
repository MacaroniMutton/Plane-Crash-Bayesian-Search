import requests
import json

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
    

