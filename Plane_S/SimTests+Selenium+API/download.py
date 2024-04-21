import requests
from bs4 import BeautifulSoup

lat_min = 12.0
lat_max = 15.0
lon_min = -50
lon_max = -25


gebco_url = f'https://download.gebco.net/'

# if search_bar != '' and search_bar is not None:
#     page = requests.get(search_bar)
    

#     for link in soup.find_all('a'):
#         link_address = link.get('href')
#         link_text = link.string
#         Link.objects.create(address=link_address, name=link_text)
    
#     links = Link.objects.all()

# if request.method == "POST":
#     for link in links:
#         link.delete()
#     links = []
response = requests.get(gebco_url)
soup = BeautifulSoup(response.text, 'html.parser')
print(soup)