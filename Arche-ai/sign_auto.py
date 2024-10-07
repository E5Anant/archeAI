from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import random

def translate_to_asl(text, user_agent=None, headless=False, target_language="ENGLISH", target_region="AMERICAN"):
    options = Options()
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        # ... add more user agents
    ]

    if headless:
        options.add_argument("--headless=new")

    if user_agent is None:  
        options.add_argument(f"user-agent={random.choice(user_agents)}")
    else:
        options.add_argument(f"user-agent={user_agent}")

    # Let webdriver_manager handle the driver:
    driver = webdriver.Chrome(options=options)  
    driver.get("https://sign.mt") 

    try:
        # 1. Select Target Language
        language_dropdown = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "/html/body/app-root/ion-app/ion-router-outlet/app-main/ion-tabs/div/ion-router-outlet/app-translate/app-translate-desktop/ion-content/div/app-drop-pose-file/div/div/app-language-selectors/app-language-selector[2]/ion-button[1]//button")) # Replace with actual ID
        )
        language_select = Select(language_dropdown)
        language_select.select_by_visible_text(target_language)

        # 2. Select Target Region (if applicable)
        region_dropdown = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "/html/body/app-root/ion-app/ion-router-outlet/app-main/ion-tabs/div/ion-router-outlet/app-translate/app-translate-desktop/ion-content/div/app-drop-pose-file/div/div/app-language-selectors/app-language-selector[1]/ion-button[1]//button/span"))  # Replace with actual ID
        )
        region_select = Select(region_dropdown)
        region_select.select_by_visible_text(target_region)
        # 1. Find the input field and enter text 
        input_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "/html/body/app-root/ion-app/ion-router-outlet/app-main/ion-tabs/div/ion-router-outlet/app-translate/app-translate-desktop/ion-content/div/app-drop-pose-file/div/div/div/app-spoken-to-signed/app-spoken-language-input/div/app-desktop-textarea/textarea")) # Replace if needed
        )
        input_field.clear()  # Clear any default text in the field
        time.sleep(1)
        
        input_field.send_keys(text)
        time.sleep(3)
        input_field.send_keys(Keys.RETURN)  # Simulate pressing Enter, might be needed to trigger translation 

        # 2. Wait for the translation to load (adjust the sleep if necessary)
        time.sleep(5) # This is not ideal but might be necessary. Observe the website's behavior. 

        # 3. Retrieve the translation output
        #    - This depends on how the website displays the ASL translation
        #    - Options include:
        #       - Extracting text if there's a text-based representation.
        #       - Getting the URL of the video or images if those are used.

        # Example: Getting the source URL of the first video element (adapt if necessary)
        video_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '/html/body/app-root/ion-app/ion-router-outlet/app-main/ion-tabs/div/ion-router-outlet/app-translate/app-translate-desktop/ion-content/div/app-drop-pose-file/div/div/div/app-spoken-to-signed/app-signed-language-output/video')) 
        )
        video_url = video_element.get_attribute('src')

        return video_url 

    finally:
        driver.quit()

if __name__ == "__main__":
    text_to_translate = "Hello, how are you doing today?"
    asl_translation_url = translate_to_asl(text_to_translate, headless=False, user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    print(f"ASL Translation URL: {asl_translation_url}")