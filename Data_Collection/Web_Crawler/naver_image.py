###### https://blog.himion.com/175 참고
'''
* 네이버 이미지 가져오기 (24.07.26)
'''

import datetime
import os
import time
import urllib
from itertools import product

from selenium import webdriver
from selenium.webdriver.common.by import By

# 검색어 합성
# ['91년식 중고차', ... , '05년식 중고차']
# ['1991년식 중고차', ... , '2005년식 중고차']
# ['1991 중고차', ... , '2005 중고차']
# 91년식, 92년식, ~~ 06년식 <-- 이때까지는 녹판 있었음
# ['녹색번호판'] + 기간
# 기간 + 첫차 중고차 세차 새차 차 자가용
# []['1991','1992','1993'] ['년식', '생산', ''] ['중고차','판매완료'] ['녹색번호판']

option = 1
if option == 1:
    years = [f'{i % 100}년식' for i in range(1991, 2007)]
    keywords = list(product(years, ['', ' 중고차', ' 판매완료']))
elif option == 2:
    years = [str(i) for i in range(1991, 2007)]
    keywords = list(product(years,['', '년식'],[' 중고차']))

item_list = [''.join(k) for k in keywords]  # 1번
FOLDER = 'naver'  # 2번
IMG_XPATH = '/html/body/div[4]/div/div/div[1]/div[2]/div[1]/img'


def main():
    start = check_start()  # 시간 측정 시작
    driver = webdriver.Chrome()

    for searchItem in item_list:
        saveDir = makeFolder(searchItem)

        url = makeUrl(searchItem)  # 검색할 url 가져와서
        driver.get(url)  # 이미지 검색으로 가서
        maximizeWindow(driver)  # 창최대화
        scrollToEnd(driver)

        forbiddenCount = saveImgs(driver, saveDir, start)  # 모든 상세 이미지 src들을 가져온다
        sec = check_time(start)
        print(f'실패수{str(forbiddenCount)}, {sec}, {datetime.datetime.now().time()}')
    time.sleep(10)
    driver.quit()


# 이미지 검색 url 만들기
def makeUrl(searchItem):
    url = 'https://search.naver.com/search.naver'
    params = {
        'where': 'image',
        'sm': 'tab_jum',
        'query': searchItem
    }
    url = url + '?' + urllib.parse.urlencode(params)
    return url


# 폴더 생성
def makeFolder(searchItem):
    saveDir = os.path.join(os.getcwd(), 'data', f'{FOLDER}_{searchItem}')
    try:
        if not (os.path.isdir(saveDir)):  # 해당 폴더가 없다면
            os.makedirs(os.path.join(saveDir))  # 만들어라
        return saveDir
    except OSError as e:
        print(e + '폴더 생성 실패')


# 창 최대화
def maximizeWindow(driver):
    driver.maximize_window()


# 모든 이미지 목록을 가져오기 위해 무한 스크롤 다운
def scrollToEnd(driver):
    prev_height = driver.execute_script('return document.body.scrollHeight')
    print(f'prev_height: {prev_height}')

    while True:
        time.sleep(1)  # 네이버는 sleep없이 이동할 경우 무한로딩에 걸린다.
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')
        time.sleep(3)

        cur_height = driver.execute_script('return document.body.scrollHeight')
        print(f'cur_height: {cur_height}')
        if cur_height == prev_height:
            print('높이가 같아짐')
            break
        prev_height = cur_height

    # 페이지를 모두 로딩한 후에는 최상단으로 다시 올라가기
    driver.execute_script('window.scrollTo(0, 0)')


# 모든 이미지들을 저장한다
def saveImgs(driver, saveDir, start):
    time.sleep(1)
    forbiddenCount = 0
    imgs = driver.find_elements(By.CSS_SELECTOR, '._fe_image_tab_content_thumbnail_image')

    print('imgs')
    print(imgs)
    srcList = []
    img_count = len(imgs)
    print(f'전체 이미지수 : {img_count}')
    # 하나씩 클릭해가며 저장
    for imgNum, img in enumerate(imgs):  # imgNum에 이미지번호가 0부터 들어간다
        try:
            img.click()
            time.sleep(2)

            # 아래의 xPath는 자주 바뀌는 것 같다. 나머지는 고정인거 같으니 이것만 가끔 확인해주자
            bigImg = driver.find_element(By.XPATH, IMG_XPATH)
            src = bigImg.get_attribute('src')
            src = urllib.parse.unquote(src.split('&type')[0].split('?src=')[-1])  # hrkim 추가한 부분
            urllib.request.urlretrieve(src, saveDir + '/' + str(imgNum) + '.jpg')
            sec = check_time(start)
            print(f'{imgNum + 1}/{img_count} saved {sec}')

        except Exception as e:
            print(e)
            forbiddenCount += 1  # 저장 실패한 개수. forbidden이나 파일에러도 꽤 많다
            continue
    return forbiddenCount


# 시간 측정
def check_start():
    start_time = time.time()
    print("Start! now.." + str(start_time))
    return start_time


def check_time(start):
    end = time.time()
    during = end - start
    sec = str(datetime.timedelta(seconds=during)).split('.')[0]
    return sec


main()
