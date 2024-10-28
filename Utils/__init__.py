import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from bidict import bidict

# 영한변환표 v1.0
bd_eng2kor_v1p0 = bidict(dict(
    ZA='강원', ZB='경기', ZC='경남', ZD='경북', ZE='광주', ZF='대구', ZG='대전', ZH='부산',
    ZI='서울', ZJ='울산', ZK='인천', ZL='전남', ZM='전북', ZN='제주', ZO='충남', ZP='충북',
    WA='구', WB='누', WC='두', WD='루', WE='무', WF='부', WG='수', WH='우', WI='주',
    WJ='허', WK='하', WL='호', ZU='배', ZV='육', ZQ='바', ZR='사', ZS='아', ZT='자',
    A='가', B='나', C='다', D='라', E='마', F='거', G='너', H='더', I='러', J='머', K='버', L='서', M='어', N='저',
    P='고', Q='노', R='도', S='로', T='모', U='보', V='소', X='오', Y='조', Z='세종'))

# 영한변환표 v1.3
bd_eng2kor_v1p3 = bidict(
    {'AA': '가', 'AB': '나', 'AC': '다', 'AD': '라', 'AE': '마', 'AF': '거', 'AG': '너', 'AH': '더', 'AI': '러',
     'AJ': '머', 'AK': '버', 'AL': '서', 'AM': '어', 'AN': '저', 'BA': '고', 'BB': '노', 'BC': '도', 'BD': '로',
     'BE': '모', 'BF': '보', 'BG': '소', 'BH': '오', 'BI': '조', 'BJ': '구', 'BK': '두', 'BL': '무', 'BM': '수',
     'BN': '누', 'BO': '루', 'BP': '부', 'BQ': '우', 'BR': '주', 'HA': '하', 'HB': '허', 'HC': '호', 'TA': '바',
     'TB': '사', 'TC': '아', 'TD': '자', 'TE': '배', 'MA': '육', 'MB': '해', 'MC': '공', 'MD': '국', 'ME': '합',
     'TF': '차', 'TG': '카', 'TH': '타', 'TI': '파', 'DA': '강', 'DB': '경', 'DC': '광', 'DD': '대', 'DE': '세',
     'DF': '울', 'DG': '인', 'DH': '전', 'DI': '제', 'DJ': '충', 'DK': '원', 'DL': '남', 'DM': '북', 'DN': '산',
     'DO': '종', 'DP': '천', 'CA': '외', 'CB': '영', 'CC': '준', 'CD': '협', 'CE': '교', 'CF': '기', 'CG': '정',
     'CH': '표'})


def trans_eng2kor_v1p3(list_of_txt: list):
    retval = []
    for chs in list_of_txt:
        if not chs.isdigit():
            for f, t in bd_eng2kor_v1p3.items():
                chs = chs.replace(f, t)
        retval.append(chs)
    return retval


def imread_uni(filename, flags=cv2.IMREAD_COLOR):
    """
    경로에 유니코드가 섞여있으면 이 함수를 사용하라.\n
    cv2.imread로는 이미지를 불러올 수 없다.
    :param filename: jpg, png, bmp 등 이미지 파일의 절대경로
    :param flags: cv2.IMREAD_COLOR(default), cv.IMREAD_GRAYSCALE, cv.IMREAD_UNCHANGED, ..., 참고(https://docs.opencv.org/4.10.0/d8/d6a/group__imgcodecs__flags.html#ga61d9b0126a3e57d9277ac48327799c80)
    :return:
    """
    img_temp = np.fromfile(filename, np.uint8)
    return cv2.imdecode(img_temp, flags)


# GPT가 만듦
def add_text_with_background(image, text, font_path="malgunbd.ttf", font_size=20,
                             font_color=(0, 255, 0), bg_color=(0, 0, 0),
                             position=(50, 50), padding=5):
    """
    OpenCV 이미지에 한글 텍스트와 배경을 추가한 후 다시 OpenCV 이미지로 반환하는 함수

    :param image: OpenCV 컬러 이미지 (numpy 배열)
    :param text: 추가할 텍스트 (한글 가능)
    :param font_path: 사용할 폰트 경로 (기본값: "NanumGothic.ttf")
    :param font_size: 텍스트 크기 (기본값: 20)
    :param font_color: 텍스트 색상 (기본값: 초록색 (0, 255, 0))
    :param bg_color: 텍스트 배경 색상 (기본값: 검은색 (0, 0, 0))
    :param position: 텍스트가 추가될 위치 (기본값: (50, 50))
    :param padding: 텍스트 배경에 추가할 패딩 (기본값: 5)
    :return: 텍스트와 배경이 추가된 OpenCV 컬러 이미지
    """

    # OpenCV 이미지를 PIL 이미지로 변환
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)

    # 폰트 설정
    unicode_font = ImageFont.truetype(font=font_path, size=font_size)

    # 텍스트의 크기를 계산 (텍스트 경계를 반환)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=unicode_font)
    text_width = right - left
    text_height = bottom - top

    # 텍스트 배경 사각형 좌표 계산
    x, y = position
    background_left = x - padding
    background_top = y - padding + font_size // 4  # magic number
    background_right = x + text_width + padding
    background_bottom = y + text_height + padding + font_size // 4  # magic number

    # 배경 그리기 (사각형)
    draw.rectangle([background_left, background_top, background_right, background_bottom], fill=bg_color)

    # 텍스트 그리기
    draw.text((x, y), text, fill=font_color, font=unicode_font)

    # 다시 OpenCV 이미지로 변환
    result_image = np.uint8(img_pil)

    return result_image
