from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import shutil

# Configure headless chrome
options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

try:
    # Open NSE FnO Bhavcopy page (or homepage)
    driver.get("https://www.nseindia.com/market-data/live-equity-market")
    
    # Wait for page load and cookie generation
    time.sleep(5)  

    # Now directly open download URL
    url = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_20230731_F_0000.csv.zip"
    driver.get(url)
    
    # Wait to download completes (Chrome default location)
    time.sleep(10)
finally:
    driver.quit()
