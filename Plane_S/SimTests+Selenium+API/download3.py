from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

download_path = "C:\\Users\\Manan Kher\\OneDrive\\Documents\\MiniProject"


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


wait = WebDriverWait(web, 10)  # Initialize WebDriverWait

time.sleep(2)
# Wait for elements to be clickable
north = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="coordinates-card"]/div[2]/div[1]/div[1]/div/input')))
north.clear()
north.send_keys(10)

south = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="coordinates-card"]/div[2]/div[3]/div[2]/div/input')))
south.clear()
south.send_keys("-10")  # Input negative value as a string

east = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="coordinates-card"]/div[2]/div[2]/div[1]/div/input')))
east.clear()
east.send_keys(10)

west = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="coordinates-card"]/div[2]/div[2]/div[3]/div/input')))
west.clear()
west.send_keys(20)

inp = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="gridCheck"]')))
inp.click()

add_basket = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="sidebar-add-to-basket"]')))
add_basket.click()
time.sleep(5)

view_basket = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="data-selection-card"]/div[3]/button[2]')))
view_basket.click()

download = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="basket-download-button"]')))
download.click()
time.sleep(15)
