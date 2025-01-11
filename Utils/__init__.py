import json
import os
import re
from datetime import datetime, timezone, timedelta
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement

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

kor_complete_form = {
    'P1': ['가', '나', '다', '라', '마', '거', '너', '더', '러', '머', '버', '서', '어', '저', '고', '노', '도', '로', '모', '보',
           '소', '오', '조', '구', '누', '두', '루', '무', '부', '수', '우', '주', '하', '허', '호', '육', '해', '공', '국', '합'],
    'P2': ['가', '나', '다', '라', '마', '거', '너', '더', '러', '머', '버', '서', '어', '저', '고', '노', '도', '로', '모', '보',
           '소', '오', '조', '구', '누', '두', '루', '무', '부', '수', '우', '주', '허'],
    'P3': ['아', '바', '사', '자'],
    'P3prov': ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주'],
    'P4': ['아', '바', '사', '자'],
    'P4prov': ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주'],
    'P5': ['가', '나', '다', '라', '마', '거', '너', '더', '러', '머', '버', '서', '어', '저', '고', '노', '도', '로', '모', '보',
           '소', '오', '조', '구', '누', '두', '루', '무', '부', '수', '우', '주'],
    'P5prov': ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주'],
    'P6': ['가', '나', '다', '라', '마', '거', '너', '더', '러', '머', '버', '서', '어', '저', '고', '노', '도', '로', '모', '보',
           '소', '오', '조', '구', '누', '두', '루', '무', '부', '수', '우', '주']
}

colors = [(128, 0, 255),  # Rose
          (0, 0, 255),  # Red
          (0, 128, 255),  # Orange
          (0, 255, 255),  # Yellow
          (0, 255, 0),  # Green
          (255, 255, 0),  # Cyan
          (255, 0, 0),  # Blue
          (128, 0, 255),  # Violet
          (255, 0, 255),  # Magenta
          ]


def trans_eng2kor_v1p3(list_of_txt: list):
    retval = []
    for chs in list_of_txt:
        if not chs.isdigit():
            for f, t in bd_eng2kor_v1p3.items():
                chs = chs.replace(f, t)
        retval.append(chs)
    return retval


### 변환표
bd_chn2num = bidict({'京': '00',  # Beijing
                     '津': '01',  # Tianjin
                     '冀': '02',  # Hebei
                     '晋': '03',  # Shanxi
                     '蒙': '04',  # Inner Mongolia
                     '辽': '05',  # Liaoning
                     '吉': '06',  # Jilin
                     '黑': '07',  # Heilongjiang
                     '沪': '08',  # Shanghai
                     '苏': '09',  # Jiangsu
                     '浙': '10',  # Zhejiang
                     '皖': '11',  # Anhui
                     '闽': '12',  # Fujian
                     '赣': '13',  # Jiangxi
                     '鲁': '14',  # Shandong
                     '豫': '15',  # Henan
                     '鄂': '16',  # Hubei
                     '湘': '17',  # Hunan
                     '粤': '18',  # Guangdong
                     '桂': '19',  # Guangxi
                     '琼': '20',  # Hainan
                     '渝': '21',  # Chongqing
                     '川': '22',  # Sichuan
                     '贵': '23',  # Guizhou
                     '云': '24',  # Yunnan
                     '藏': '25',  # Xizang Tibetan Autonomous Region
                     '陕': '26',  # Shaanxi
                     '甘': '27',  # Gansu
                     '青': '28',  # Qinghai
                     '宁': '29',  # Ningxia
                     '新': '30',  # Xinjiang
                     '港': '31',  # Hong Kong (suffix)
                     '澳': '32',  # Macau (suffix)
                     })


def imread_uni(filename, flags=cv2.IMREAD_COLOR):
    """
    경로에 유니코드가 섞여있으면 이 함수를 사용하라.\n
    cv2.imread로는 한글 경로 파일을 불러올 수 없다.
    :param filename: jpg, png, bmp 등 이미지 파일의 절대경로
    :param flags: cv2.IMREAD_COLOR(default), cv.IMREAD_GRAYSCALE, cv.IMREAD_UNCHANGED, ..., 참고(https://docs.opencv.org/4.10.0/d8/d6a/group__imgcodecs__flags.html#ga61d9b0126a3e57d9277ac48327799c80)
    :return:
    """
    img_temp = np.fromfile(filename, np.uint8)
    return cv2.imdecode(img_temp, flags)


def imwrite_uni(filename, cv_img):
    """
    경로에 유니코드가 섞여있으면 이 함수를 사용하라.\n
    cv2.imwrite로는 한글 경로 파일을 저장할 수 없다.
    :param filename: jpg, png, bmp 등 저장하고자 하는 절대경로
    :param cv_img: opencv(numpy) 이미지 배열
    :return: 없음
    """
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(cv_img)
    im.save(filename)


# GPT가 만듦
def add_text_with_background(image, text, font_path="NanumGothicCoding-Bold.ttf", font_size=20,
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
    try:
        unicode_font = ImageFont.truetype(font=font_path, size=font_size)
    except OSError:
        font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), font_path)  # ttf 파일의 절대경로로 변경
        unicode_font = ImageFont.truetype(font=font_path, size=font_size)

    # 텍스트의 크기를 계산 (텍스트 경계를 반환)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=unicode_font)
    text_width = right - left
    text_height = bottom - top

    # 텍스트 배경 사각형 좌표 계산
    x, y = position
    background_left = x - padding
    background_top = y - padding
    background_right = x + text_width + padding
    background_bottom = y + text_height + padding

    # 배경 그리기 (사각형)
    draw.rectangle([background_left, background_top, background_right, background_bottom], fill=bg_color)

    # 텍스트 그리기
    draw.text((x, y), text, fill=font_color, font=unicode_font)

    # 다시 OpenCV 이미지로 변환
    result_image = np.uint8(img_pil)

    return result_image


def xywh2xyxy(xywh):
    assert len(xywh) == 4
    xyxy = [xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]]
    return xyxy


def xyxy2xywh(xyxy):
    assert len(xyxy) == 4
    xywh = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
    return xywh


def xywh2cxcywh(xywh):
    assert len(xywh) == 4
    w, h = xywh[2], xywh[3]
    cx = xywh[0] + w / 2
    cy = xywh[1] + h / 2
    cxcywh = [cx, cy, w, h]
    return cxcywh


def cxcywh2xywh(cxcywh):
    assert len(cxcywh) == 4
    w, h = cxcywh[2], cxcywh[3]
    x = cxcywh[0] - w / 2
    y = cxcywh[1] - h / 2
    xywh = [x, y, w, h]
    return xywh


def cxcywh2cxcysfar(cxcywh):
    assert len(cxcywh) == 4
    w, h = cxcywh[2], cxcywh[3]
    sf = w * h
    ar = w / h
    cxcysfar = [cxcywh[0], cxcywh[1], sf, ar]
    return cxcysfar


def cxcysfar2cxcywh(cxcysfar):
    assert len(cxcysfar) == 4
    sf, ar = cxcysfar[2], cxcysfar[3]
    w = (sf * ar) ** 0.5
    h = sf / w
    cxcywh = [cxcysfar[0], cxcysfar[1], w, h]
    return cxcywh


def save_json(json_path, shapes, imagePath, imageHeight, imageWidth):
    data = dict(
        version="5.5.0",  # 버전 통일
        flags={},
        shapes=shapes,
        imagePath=imagePath,
        imageData=None,
        imageHeight=imageHeight,
        imageWidth=imageWidth,
    )
    '''
    {
      "version": "5.5.0",
      "flags": {},
      "shapes": [
        {
          "label": "P3_서울71바8669",
          "points": [
            [
              408.0,
              424.0
            ],
            [
              707.0,
              593.0
            ]
          ],
          "group_id": null,
          "description": "",
          "shape_type": "rectangle",
          "flags": {},
          "mask": null
        },
        {
          "label": "P3_서울71바8669",
          "points": [
            [
              412.66490765171505,
              438.2585751978892
            ],
            [
              679.155672823219,
              428.49604221635883
            ],
            [
              692.6121372031662,
              586.0158311345647
            ],
            [
              429.02374670184696,
              589.1820580474933
            ]
          ],
          "group_id": null,
          "description": "",
          "shape_type": "polygon",
          "flags": {},
          "mask": null
        }
      ],
      "imagePath": "in_L01_0_20161125_182519_182715_438_서울71바8669.jpg",
      "imageData": null,
      "imageHeight": 1200,
      "imageWidth": 1600
    }
    '''
    save_path = os.path.join(json_path, imagePath)[:-4]
    try:
        with open(save_path + '.json', "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(e)
        print(json_path)


def save_xml(xml_path, xyxys, labels):
    '''
    <annotation>
    <filename></filename>
    <object>
        <name>P3_서울71바8669</name>
        <bndbox>
            <xmin>408</xmin>
            <ymin>424</ymin>
            <xmax>707</xmax>
            <ymax>593</ymax>
        </bndbox>
    </object>
    </annotation>
    '''
    assert len(xyxys) == len(labels)
    root = Element('annotation')
    SubElement(root, 'filename')
    for i, xyxy in enumerate(xyxys):
        obj = SubElement(root, 'object')
        SubElement(obj, 'name').text = labels[i]
        bndbox = SubElement(obj, 'bndbox')
        SubElement(bndbox, 'xmin').text = str(xyxy[0])
        SubElement(bndbox, 'ymin').text = str(xyxy[1])
        SubElement(bndbox, 'xmax').text = str(xyxy[2])
        SubElement(bndbox, 'ymax').text = str(xyxy[3])

    with open(xml_path, 'w', encoding='utf-8') as f:
        ElementTree.indent(root)
        print(ElementTree.tostring(root, encoding='utf-8').decode(), file=f)
        # f.write(ElementTree.tostring(root,encoding='utf-8').decode())


def iou(bb1, bb2):
    """
    intersection over union 계산
    :param bb1: bounding box를 의미한다. len==4인 list로, [x1, y1, x2, y2] 의 순서를 갖는다.
    :param bb2: bb1과 같다.
    :return: intersection over union 값
    """
    xx1 = np.maximum(bb1[0], bb2[0])
    yy1 = np.maximum(bb1[1], bb2[1])
    xx2 = np.minimum(bb1[2], bb2[2])
    yy2 = np.minimum(bb1[3], bb2[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb1[2] - bb1[0]) * (bb1[3] - bb1[1]) +
              (bb2[2] - bb2[0]) * (bb2[3] - bb2[1]) - wh)
    return o


def iou_4corner(b1, b2): # 4꼭지점을 이용한 iou
    if 'BBox' in str(b1.__class__):
        b1 = np.float32([(b1.x, b1.y), (b1.x + b1.w, b1.y), (b1.x + b1.w, b1.y + b1.h), (b1.x, b1.y + b1.h)])
    else:
        b1 = np.float32([b1[0], b1[1], b1[2], b1[3]])
    if 'BBox' in str(b2.__class__):
        b2 = np.float32([(b2.x, b2.y), (b2.x + b2.w, b2.y), (b2.x + b2.w, b2.y + b2.h), (b2.x, b2.y + b2.h)])
    else:
        b2 = np.float32([b2[0], b2[1], b2[2], b2[3]])

    inter_area, _ = cv2.intersectConvexConvex(b1, b2)
    area1 = cv2.contourArea(b1)
    area2 = cv2.contourArea(b2)
    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def plate_number_tokenizer(plate_number='서울12가3456'):
    digits = re.findall('\\d+', plate_number)
    koreans = re.findall('[가-힣]+', plate_number)
    digit_2, digit_4 = digits
    kor_mid = koreans.pop(-1)
    kor_prov = ''.join(koreans)
    return kor_prov, digit_2, kor_mid, digit_4


KST = timezone(timedelta(hours=9))


def get_pretty_datetime(add_ms=True, add_TZ=False):
    retval = datetime.now(KST).strftime('%Y%m%d_%H%M%S')
    if add_ms:
        retval += datetime.now(KST).strftime('_%f')[:-3]
    if add_TZ:
        retval += datetime.now(KST).strftime('%z')
    return retval
