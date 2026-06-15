"""Visit the Streamlit app with a real headless browser and wake it if asleep."""
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

URL = os.environ.get("APP_URL", "https://futurestars.streamlit.app/")

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--no-sandbox")
opts.add_argument("--disable-dev-shm-usage")
opts.add_argument("--window-size=1280,900")

driver = webdriver.Chrome(options=opts)
try:
    driver.get(URL)
    time.sleep(8)  # let the page load
    # If the app is asleep, Streamlit shows a "Yes, get this app back up!" button.
    clicked = False
    for btn in driver.find_elements(By.TAG_NAME, "button"):
        label = (btn.text or "").lower()
        if "get this app back up" in label or "back up" in label:
            btn.click()
            clicked = True
            print("App was asleep -> clicked wake button.")
            time.sleep(30)  # wait for it to boot
            break
    print(f"Visited {URL} | title='{driver.title}' | woke_up={clicked}")
finally:
    driver.quit()
