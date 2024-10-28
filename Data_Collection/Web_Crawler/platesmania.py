import subprocess
from time import sleep

import pyautogui
import pyperclip
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# 10개 다 긁기
pick_car = ['body > div.wrapper > div.container.content > div > div.col-md-9 > div:nth-child(5) > div:nth-child(1) > div > div.panel-body > div:nth-child(1) > a > img',
            'body > div.wrapper > div.container.content > div > div.col-md-9 > div:nth-child(5) > div:nth-child(2) > div > div.panel-body > div:nth-child(1) > a > img',
            'body > div.wrapper > div.container.content > div > div.col-md-9 > div:nth-child(6) > div:nth-child(1) > div > div.panel-body > div:nth-child(1) > a > img',
            'body > div.wrapper > div.container.content > div > div.col-md-9 > div:nth-child(6) > div:nth-child(3) > div > div.panel-body > div:nth-child(1) > a > img',
            'body > div.wrapper > div.container.content > div > div.col-md-9 > div:nth-child(7) > div:nth-child(1) > div > div.panel-body > div:nth-child(1) > a > img',
            'body > div.wrapper > div.container.content > div > div.col-md-9 > div:nth-child(7) > div:nth-child(2) > div > div.panel-body > div:nth-child(1) > a > img',
            'body > div.wrapper > div.container.content > div > div.col-md-9 > div:nth-child(9) > div:nth-child(1) > div > div.panel-body > div:nth-child(1) > a > img',
            'body > div.wrapper > div.container.content > div > div.col-md-9 > div:nth-child(9) > div:nth-child(3) > div > div.panel-body > div:nth-child(1) > a > img',
            'body > div.wrapper > div.container.content > div > div.col-md-9 > div:nth-child(10) > div:nth-child(1) > div > div.panel-body > div:nth-child(1) > a > img',
            'body > div.wrapper > div.container.content > div > div.col-md-9 > div:nth-child(10) > div:nth-child(2) > div > div.panel-body > div:nth-child(1) > a > img']
next_page = 'body > div.wrapper > div.container.content > div > div.col-md-9 > div:nth-child(2) > ul > li:nth-child(12) > a'
# 1개만 긁기
# pick_car = ['body > div.wrapper > div.container.content > div > div.col-md-9 > div:nth-child(10) > div:nth-child(2) > div > div.panel-body > div:nth-child(1) > a > img']

url = 'https://platesmania.com/kr/gallery.php?gal=kr&fon=1'  # 폰트 1페이지


# url = 'https://platesmania.com/kr/gallery.php?&fon=5&start=8' # 특정 페이지

# css selector 받아오기
def wait_for_elem(selector):
    try:
        elem = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
    except:
        elem = None
    return elem


# 크롬 브라우져 켜기
subprocess.Popen(r'C:\Program Files\Google\Chrome\Application\chrome.exe --remote-debugging-port=9222 --user-data-dir="C:\chromeCookie"')
option = Options()
option.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=option)
# driver.maximize_window()

# url 진입
aaa = wait_for_elem('# input')
driver.get(url)

# 차량 선택
while True:
    for i in pick_car:
        # i번째 차량 선택
        aaa = wait_for_elem(i)
        aaa.click()

        # 차량 번호 복사
        aaa = wait_for_elem('body > div.wrapper > div.breadcrumbs > div > div > h1')
        pyperclip.copy(aaa.text)

        # pyautogui.moveTo(945, 650) # x 100, y 200 위치로 바로 이동
        # pyautogui.dragTo(1200, 650, 1)
        # sleep(0.2)
        # pyautogui.hotkey('ctrl', 'c')
        # sleep(0.2)

        # 사진 클릭
        aaa = wait_for_elem('body > div.wrapper > div.container.content > div:nth-child(1) > div.col-md-6.col-sm-7 > div > div.panel-body > div:nth-child(1) > a > img')
        aaa.click()
        sleep(1.5)

        # 사진 저장
        screen_width, screen_height = pyautogui.size()
        pyautogui.moveTo(screen_width // 2, screen_height // 2)  # 화면 중앙 위치로 이동
        sleep(0.5)
        pyautogui.click(button='right')
        sleep(0.5)
        pyautogui.write('v')
        sleep(0.5)
        pyautogui.press('right', presses=1)
        sleep(0.2)
        pyautogui.write('_')
        sleep(0.2)
        pyautogui.hotkey('ctrl', 'v')
        sleep(0.2)
        pyautogui.press('enter')
        sleep(0.2)

        # 중복시 덮어쓰기
        pyautogui.write('y')
        sleep(0.2)
        pyautogui.hotkey('ctrl', 'w')
        sleep(0.3)

        # 뒤로가기
        driver.back()

    # 다음 페이지가기
    aaa = wait_for_elem(next_page)
    if aaa == None:
        next_page = 'body > div.wrapper > div.container.content > div > div.col-md-9 > div:nth-child(2) > ul > li:nth-child(8) > a'
        aaa = wait_for_elem(next_page)
    aaa.click()
