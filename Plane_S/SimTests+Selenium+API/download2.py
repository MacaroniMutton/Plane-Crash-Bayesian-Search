import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
import time

download_path = "C:\\Users\\Manan Kher\\OneDrive\\Documents\\MINI_PROJECT\\Plane-Crash-Bayesian-Search\\Plane_S\\SimTests+Selenium+API"

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

north = web.find_element('xpath','//*[@id="coordinates-card"]/div[2]/div[1]/div[1]/div/input')
north.clear()
north.send_keys(Keys.CONTROL + "a")
north.send_keys("12")


south = web.find_element('xpath','//*[@id="coordinates-card"]/div[2]/div[3]/div[2]/div/input')
south.clear()
south.send_keys(Keys.CONTROL + "a")
south.send_keys("-10")


east = web.find_element('xpath','//*[@id="coordinates-card"]/div[2]/div[2]/div[1]/div/input')
east.clear()
east.send_keys(Keys.CONTROL + "a")
east.send_keys("-10")


west = web.find_element('xpath','//*[@id="coordinates-card"]/div[2]/div[2]/div[3]/div/input')
west.clear()
west.send_keys(Keys.CONTROL + "a")
west.send_keys("-1")



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



