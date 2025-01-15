import os
from pathlib import Path
from time import time

import cv2
import numpy as np

from Utils import imread_uni, plate_number_tokenizer, bd_eng2kor_v1p3, trans_kor2eng_v1p3


class Graphical_Model_Generator_KOR:
    plate_wh = {
        'P1-1': (520, 110), 'P1-2': (520, 110), 'P1-3': (520, 110), 'P1-4': (520, 110),
        'P2': (440, 200), 'P3': (440, 220), 'P4': (520, 110), 'P5': (335, 170), 'P6': (335, 170),
    }
    char_xywh = {
        'P1-1': [(44, 13.5, 56, 83), (100, 13.5, 56, 83), (156, 13.5, 96, 83), (252, 13.5, 56, 83), (308, 13.5, 56, 83), (364, 13.5, 56, 83), (420, 13.5, 56, 83)],
        'P1-2': [(44, 13.5, 56, 83), (100, 13.5, 56, 83), (156, 13.5, 96, 83), (252, 13.5, 56, 83), (308, 13.5, 56, 83), (364, 13.5, 56, 83), (420, 13.5, 56, 83)],
        'P1-3': [(30, 13.5, 55, 83), (85, 13.5, 55, 83), (140, 13.5, 55, 83), (195, 13.5, 75, 83), (270, 13.5, 55, 83), (325, 13.5, 55, 83), (380, 13.5, 55, 83), (435, 13.5, 55, 83)],
        'P1-4': [(65, 12.5, 50, 85), (115, 12.5, 50, 85), (165, 12.5, 50, 85), (215, 12.5, 85, 85), (300, 12.5, 50, 85), (350, 12.5, 50, 85), (400, 12.5, 50, 85), (450, 12.5, 50, 85)],
        'P2': [(11, 60, 59, 105), (70, 60, 59, 105), (129, 60, 64, 105), (193, 60, 59, 105), (252, 60, 59, 105), (311, 60, 59, 105), (370, 60, 59, 105)],
        'P3': [(107, 11, 126, 61), (233, 11, 50, 61), (283, 11, 50, 61), (19, 84, 89, 116), (108, 84, 78, 116), (186, 84, 78, 116), (264, 84, 78, 116), (342, 84, 78, 116)],
        'P4': [(32, 13.5, 55, 83), (87, 13.5, 55, 83), (142, 13.5, 55, 83), (197, 13.5, 71, 83), (268, 13.5, 55, 83), (323, 13.5, 55, 83), (378, 13.5, 55, 83), (433, 13.5, 55, 83)],
        'P5': [(90, 14, 79, 36), (178, 14, 30, 40), (215, 14, 30, 40), (22, 60, 60, 60), (105, 60, 45, 90), (161, 60, 45, 90), (217, 60, 45, 90), (273, 60, 45, 90)],
        'P6': [(90, 13, 45, 37), (145, 13, 45, 37), (200, 13, 45, 37), (15, 64, 65, 92), (95, 64, 65, 92), (175, 64, 65, 92), (255, 64, 65, 92)],
    }

    def __init__(self, base_path='BetaType/korean_LP/', scale_up=2):
        self.base_path = os.path.join(os.path.dirname(__file__), base_path)
        self.support_ptypes = ['P1-1', 'P1-2', 'P1-3', 'P1-4', 'P2', 'P3', 'P4', 'P5', 'P6']
        self.plate_wh_up = {t: tuple(map(lambda x: int(x * scale_up), self.plate_wh[t])) for t in self.plate_wh}
        self.char_xywh_up = {t: [tuple(map(lambda x: int(x * scale_up), xywh)) for xywh in self.char_xywh[t]] for t in self.char_xywh}

        self.images = {t: {} for t in self.support_ptypes}
        self.__load_images()

    def __load_images(self):
        for t in self.support_ptypes:
            p_in = Path(self.base_path) / Path(t)
            for fpath in p_in.glob('**/*.png'):
                img = imread_uni(fpath, cv2.IMREAD_UNCHANGED)
                parent = fpath.parent.stem
                if parent == 'template':
                    img = cv2.resize(img, self.plate_wh_up[t])
                else:
                    gparent = fpath.parent.parent.stem
                    if gparent == 'number':
                        if '_' in parent:  # upper number [-7]
                            img = cv2.resize(img, self.char_xywh_up[t][-7][2:])
                        else:  # lower number [-1]
                            img = cv2.resize(img, self.char_xywh_up[t][-1][2:])
                    else:  # korean
                        if len(parent) == 4:  # province korean [0]
                            img = cv2.resize(img, self.char_xywh_up[t][0][2:])
                        else:  # lower korean [-5]
                            img = cv2.resize(img, self.char_xywh_up[t][-5][2:])
                self.images[t][parent] = img

    def overlay(self, bg, fg, xy):
        fh, fw = fg.shape[:2]
        x, y = map(int, xy)

        fg_rgb = fg[:, :, :3]  # RGB 채널
        alpha = fg[:, :, 3]  # alpha 채널

        bg[y:y + fh, x:x + fw, :3] = (
                alpha[:, :, np.newaxis] / 255.0 * fg_rgb +
                (1 - alpha[:, :, np.newaxis] / 255.0) * bg[y:y + fh, x:x + fw, :3]
        ).astype(np.uint8)

    def make_LP(self, plate_number, plate_type):
        assert plate_type in self.plate_wh
        assert '배' not in plate_number

        img_bg = self.images[plate_type]['template'].copy()

        img_fgs = []
        kor_prov, digit_2, kor_mid, digit_4 = plate_number_tokenizer(plate_number)

        # 지방 문자
        if len(kor_prov) > 0:
            eng = ''.join(trans_kor2eng_v1p3(kor_prov))
            img_fgs.append(self.images[plate_type][eng])

        # 가운데 문자 앞의 숫자
        ad = '_1' if '0_1' in self.images[plate_type] else ''
        for c in digit_2:
            img_fgs.append(self.images[plate_type][c + ad])

        # 가운데 문자
        eng = bd_eng2kor_v1p3.inverse[kor_mid]
        img_fgs.append(self.images[plate_type][eng])

        # 가운데 문자 뒤의 숫자
        for c in digit_4:
            img_fgs.append(self.images[plate_type][c])

        # overlay 시키기
        for i, xywh in enumerate(self.char_xywh_up[plate_type]):
            xy = xywh[:2]
            self.overlay(img_bg, img_fgs[i], xy)

        return img_bg

    def get_plate_number_area_only(self, plate_type):
        left_box = min(self.char_xywh[plate_type], key=lambda x: x[0])
        top_box = min(self.char_xywh[plate_type], key=lambda x: x[1])
        right_box = max(self.char_xywh[plate_type], key=lambda x: x[0] + x[2])
        bottom_box = max(self.char_xywh[plate_type], key=lambda x: x[1] + x[3])
        min_x = left_box[0]
        min_y = top_box[1]
        max_x = right_box[0] + right_box[2]
        max_y = bottom_box[1] + bottom_box[3]
        return min_x, min_y, max_x, max_y


if __name__ == '__main__':
    st = time()
    generator = Graphical_Model_Generator_KOR()
    print('ctor %.3fms' % ((time() - st) * 1000))

    st = time()
    img = generator.make_LP('12가3456', 'P1-1')
    print('P1-1 %.3fms' % ((time() - st) * 1000))
    cv2.imshow('img', img)
    cv2.waitKey()

    st = time()
    img = generator.make_LP('78거9012', 'P1-2')
    print('P1-2 %.3fms' % ((time() - st) * 1000))
    cv2.imshow('img', img)
    cv2.waitKey()

    st = time()
    img = generator.make_LP('356수4846', 'P1-3')  # hrkim's
    print('P1-3 %.3fms' % ((time() - st) * 1000))
    cv2.imshow('img', img)
    cv2.waitKey()

    st = time()
    img = generator.make_LP('259노7549', 'P1-4')  # 아반떼
    print('P1-4 %.3fms' % ((time() - st) * 1000))
    cv2.imshow('img', img)
    cv2.waitKey()

    st = time()
    img = generator.make_LP('98구7654', 'P2')
    print('P2 %.3fms' % ((time() - st) * 1000))
    cv2.imshow('img', img)
    cv2.waitKey()

    st = time()
    img = generator.make_LP('인천74바8282', 'P3')
    print('P3 %.3fms' % ((time() - st) * 1000))
    cv2.imshow('img', img)
    cv2.waitKey()

    st = time()
    img = generator.make_LP('대전31아4335', 'P4')
    print('P4 %.3fms' % ((time() - st) * 1000))
    cv2.imshow('img', img)
    cv2.waitKey()

    st = time()
    img = generator.make_LP('서울75마2684', 'P5')
    print('P5 %.3fms' % ((time() - st) * 1000))
    cv2.imshow('img', img)
    cv2.waitKey()

    st = time()
    img = generator.make_LP('12가3456', 'P6')
    print('P6 %.3fms' % ((time() - st) * 1000))
    cv2.imshow('img', img)
    cv2.waitKey()
