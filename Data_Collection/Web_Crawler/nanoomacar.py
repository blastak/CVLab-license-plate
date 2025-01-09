import os
import subprocess
import urllib

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from Data_Collection.Duplicate_Checker import calc_file_hash
from platesmania import wait_for_elem


# 이미지 다운로드 함수
def download_image(image_url, save_dir, plate_type, label):
    for i in range(5):
        url = image_url.replace('0.jpg', f'{i}.jpg')
        save_path = os.path.join(save_dir, f'{plate_type}_{label}_{i}.jpg')
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                h = calc_file_hash(save_path)
                os.rename(save_path, os.path.join(save_dir, f'{h}_{plate_type}_{label}.jpg'))
                print(f"Saved: {save_path}")
            else:
                print(f"Failed to download {url}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")


def euckr_to_utf8(str):
    decoded_bytes = urllib.parse.unquote_to_bytes(str)
    decoded_euc_kr = decoded_bytes.decode('euc-kr')  # euc-kr로 디코딩
    return decoded_euc_kr


# 전기차 이미지 크롤링 함수
def scrape_electric_car_images(url, plate_type, save_dir):
    subprocess.Popen(r'C:\Program Files\Google\Chrome\Application\chrome.exe --remote-debugging-port=9222 --user-data-dir="C:\chromeCookie"')
    option = Options()
    option.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=option)

    # url 진입
    a = wait_for_elem('# input')
    driver.get(url)

    n = 2
    try:
        while True:
            for i in range(2, 22):
                selector = f"#carListLayer > table > tbody > tr:nth-child({i}) > td:nth-child(2) > a > div > img"
                aa = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                image_src = aa.get_attribute("src")[:-6]
                print(f"Image src: {image_src}")

                selector = f"#carListLayer > table > tbody > tr:nth-child({i}) > td:nth-child(8) > a.btn.btn_line"
                element = driver.find_element(By.CSS_SELECTOR, selector)
                href = element.get_attribute('href').split('carNum=')[1]

                label = euckr_to_utf8(href)

                if image_src:
                    download_image(image_src, save_dir, plate_type, label)

            # 다음 페이지
            if n % 5 == 1:
                css_selector = "#paging > a:nth-child(8)"
                element = driver.find_element(By.CSS_SELECTOR, css_selector)
                driver.execute_script("arguments[0].click();", element)
            else:
                element = driver.find_element(By.LINK_TEXT, f"{n}")
                driver.execute_script("arguments[0].click();", element)
            n += 1

    except Exception as e:
        print(f"Error: {e}")
    finally:
        driver.quit()


def nanoomacar(url, plate_type=None):
    save_dir = f"./nanoomacar/{plate_type}"
    os.makedirs(save_dir, exist_ok=True)

    scrape_electric_car_images(url, plate_type, save_dir)


if __name__ == "__main__":
    url = 'https://nanoomacar.com/car/carList.html?sG=&cho=1&cGubun=&sGubun=&orderKey=&viewType=&pageSize=20&memGubun=&carGubun=&m_id=&company=&c_nameInit=&c_name=&series=&sdetail=&fuelcheck=&c_recomYN=&year1=&year2=&dFuel%5B%5D=%C0%FC%B1%E2&price1=&price2=&mileage1=&mileage2=&scolor=&boption='
    plate_type = 'P1-2'

    # plate_type 지정
    nanoomacar(url, plate_type)
    # plate_type 미지정
    # nanoomacar(url)
