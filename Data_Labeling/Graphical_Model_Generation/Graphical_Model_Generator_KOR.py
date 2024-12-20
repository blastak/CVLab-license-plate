import os
import random

import cv2
import numpy as np

from Utils import bd_eng2kor_v1p3


class Graphical_Model_Generator_KOR:
    LP_plate_wh = {
        'P1-1': (520, 110),
        'P1-2': (520, 110),
        'P1-3': (520, 110),
        'P1-4': (520, 110),
        'P2': (440, 200),
        'P3': (440, 220),
        'P4': (520, 110),
        'P5': (335, 170),
        'P6': (335, 170),
    }
    LP_char_xywh = {
        'P1-1': [(44, 13.5), (56, 83), (100, 13.5), (56, 83), (156, 13.5), (96, 83), (252, 13.5), (56, 83),  # P1-1, P1-2
                 (308, 13.5), (56, 83), (364, 13.5), (56, 83), (420, 13.5), (56, 83), ],
        'P1-2': [(44, 13.5), (56, 83), (100, 13.5), (56, 83), (156, 13.5), (96, 83), (252, 13.5), (56, 83),  # P1-1, P1-2
                 (308, 13.5), (56, 83), (364, 13.5), (56, 83), (420, 13.5), (56, 83), ],
        'P1-3': [(30, 13.5), (55, 83), (85, 13.5), (55, 83), (140, 13.5), (55, 83), (195, 13.5), (75, 83),  # P1-3
                 (270, 13.5), (55, 83), (325, 13.5), (55, 83), (380, 13.5), (55, 83), (435, 13.5), (55, 83), ],
        'P1-4': [(65, 12.5), (50, 85), (115, 12.5), (50, 85), (165, 12.5), (50, 85), (215, 12.5), (85, 85),  # P1-4
                 (300, 12.5), (50, 85), (350, 12.5), (50, 85), (400, 12.5), (50, 85), (450, 12.5), (50, 85), ],
        'P2': [(11, 60), (59, 105), (70, 60), (59, 105), (129, 60), (64, 105), (193, 60), (59, 105),  # P2
               (252, 60), (59, 105), (311, 60), (59, 105), (370, 60), (59, 105), ],
        'P3': [(107, 11), (126, 61), (233, 11), (50, 61), (283, 11), (50, 61), (19, 84), (89, 116), (108, 84), (78, 116),  # P3
               (186, 84), (78, 116), (264, 84), (78, 116), (342, 84), (78, 116), ],
        'P4': [(32, 13.5), (55, 83), (87, 13.5), (55, 83), (142, 13.5), (55, 83), (197, 13.5), (71, 83),  # P4
               (268, 13.5), (55, 83), (323, 13.5), (55, 83), (378, 13.5), (55, 83), (433, 13.5), (55, 83), ],
        'P5': [(90, 14), (79, 36), (178, 14), (30, 40), (215, 14), (30, 40), (22, 60), (60, 60),  # P5
               (105, 60), (45, 90), (161, 60), (45, 90), (217, 60), (45, 90), (273, 60), (45, 90), ],
        'P6': [(90, 13), (45, 37), (145, 13), (45, 37), (200, 13), (45, 37),  # P6
               (15, 64), (65, 92), (95, 64), (65, 92), (175, 64), (65, 92), (255, 64), (65, 92), ],
    }

    def __init__(self, base_path='BetaType/korean_LP/'):
        self.base_path = os.path.join(os.path.dirname(__file__), base_path)
        self.plate_wh = None
        self.char_xywh = None
        self.plate_wh2 = None
        self.char_xywh2 = None

    def absoluteFilePaths(self, directory):
        retval = []
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                retval.append(os.path.abspath(os.path.join(dirpath, f)))
        return retval

    def random_file_in_dir(self, directory):
        t = self.absoluteFilePaths(directory)
        return random.choice(t)

    def overlay(self, bg, fg, xy):
        fh, fw = fg.shape[:2]
        x, y = map(int, xy)

        fg_rgb = fg[:, :, :3]  # RGB 채널
        alpha = fg[:, :, 3]  # alpha 채널

        bg[y:y + fh, x:x + fw, :3] = (alpha[:, :, np.newaxis] / 255.0 * fg_rgb + (1 - alpha[:, :, np.newaxis] / 255.0) * bg[y:y + fh, x:x + fw, :3]).astype(np.uint8)

    def make_LP(self, demand_str, LP_cls):
        base_path = os.path.join(self.base_path, LP_cls)
        self.plate_wh = Graphical_Model_Generator_KOR.LP_plate_wh.get(LP_cls)
        self.char_xywh = Graphical_Model_Generator_KOR.LP_char_xywh.get(LP_cls)

        # scaling
        self.plate_wh2 = tuple(x * 2 for x in self.plate_wh)
        self.char_xywh2 = [tuple(x * 2 for x in xywh) for xywh in self.char_xywh]

        img_template = cv2.resize(cv2.imread(self.random_file_in_dir(base_path + '/template/'), cv2.IMREAD_UNCHANGED), self.plate_wh2)

        p = ''
        if LP_cls == 'P4':
            for i, ch in enumerate(demand_str):
                if i == 1:
                    continue
                if ch.isdigit():  # number
                    p = '/number/' + f'/{ch}/'
                else:  # korean
                    p = '/korean/' + f'/{bd_eng2kor_v1p3.inverse[ch]}/'
                    if i == 0:
                        p = p[:-1] + f'{bd_eng2kor_v1p3.inverse[demand_str[1]]}/'
                        i += 1
                img = cv2.resize(cv2.imread(self.random_file_in_dir(base_path + p), cv2.IMREAD_UNCHANGED), self.char_xywh2[((i - 1) * 2) + 1])
                self.overlay(img_template, img, self.char_xywh2[((i - 1) * 2)])

        elif LP_cls == 'P3' or LP_cls == 'P5':
            for i, ch in enumerate(demand_str):
                if i == 1:
                    continue
                if ch.isdigit():  # number
                    p = f'/number/{ch}'
                    if i == 2 or i == 3:  # 윗자리 숫자 예외처리
                        p += '_1/'
                else:  # korean

                    # TODO : 여기에서 '지역'이 아니거나 '아바사자배'가 아닌 것들에 대해 필터링을 할 필요가 있다. 안하면 random_file_in_dir()에서 에러

                    p = f'/korean/{bd_eng2kor_v1p3.inverse[ch]}/'
                    if i == 0:
                        p = p[:-1] + f'{bd_eng2kor_v1p3.inverse[demand_str[1]]}/'
                        i += 1
                img = cv2.resize(cv2.imread(self.random_file_in_dir(base_path + p), cv2.IMREAD_UNCHANGED), self.char_xywh2[((i - 1) * 2) + 1])
                self.overlay(img_template, img, self.char_xywh2[((i - 1) * 2)])

        elif LP_cls == 'P6':
            for i in range(7):
                if i == 0 or i == 1:
                    p = '/number/' + f'/{demand_str[i]}_1/'
                elif i == 2:
                    p = '/korean/' + f'/{bd_eng2kor_v1p3.inverse[demand_str[i]]}/'
                else:
                    p = '/number/' + f'/{demand_str[i]}/'
                img = cv2.resize(cv2.imread(self.random_file_in_dir(base_path + p), cv2.IMREAD_UNCHANGED), self.char_xywh2[(i * 2) + 1])
                self.overlay(img_template, img, self.char_xywh2[(i * 2)])

        else:
            for i, ch in enumerate(demand_str):
                if ch.isdigit():  # number
                    p = '/number/' + f'/{ch}/'
                else:  # korean
                    p = '/korean/' + f'/{bd_eng2kor_v1p3.inverse[ch]}/'
                img = cv2.resize(cv2.imread(self.random_file_in_dir(base_path + p), cv2.IMREAD_UNCHANGED), self.char_xywh2[(i * 2) + 1])
                self.overlay(img_template, img, self.char_xywh2[(i * 2)])

        img_template = cv2.cvtColor(img_template, cv2.COLOR_BGRA2BGR)
        img_template = cv2.add(img_template, np.ones_like(img_template))  # thresholding 할때 0이면 구멍이 날 수 있어서 1을 더해둠

        return img_template

    def get_text_area(self, scale=1):
        cr_x0, cr_y0 = map(int, self.char_xywh[0])
        cr_w0 = int(self.char_xywh[-2][0] + self.char_xywh[-1][0] - self.char_xywh[0][0])
        cr_h0 = int(self.char_xywh[1][1])
        return cr_x0 * scale, cr_y0 * scale, cr_w0 * scale, cr_h0 * scale


if __name__ == '__main__':
    LP_cls = 'P1-1'
    generator = Graphical_Model_Generator_KOR()
    img = generator.make_LP('12가3456', LP_cls)
    cv2.imshow('img', img)
    cv2.waitKey()

    LP_cls = 'P1-2'
    img = generator.make_LP('78거9012', LP_cls)
    cv2.imshow('img', img)
    cv2.waitKey()

    LP_cls = 'P1-3'
    img = generator.make_LP('356수4846', LP_cls)  # hrkim's
    cv2.imshow('img', img)
    cv2.waitKey()

    LP_cls = 'P1-4'
    img = generator.make_LP('259노7549', LP_cls)  # 아반떼
    cv2.imshow('img', img)
    cv2.waitKey()

    LP_cls = 'P2'
    img = generator.make_LP('98구7654', LP_cls)
    cv2.imshow('img', img)
    cv2.waitKey()

    LP_cls = 'P3'
    img = generator.make_LP('인천74바8282', LP_cls)
    cv2.imshow('img', img)
    cv2.waitKey()

    LP_cls = 'P4'
    img = generator.make_LP('대전31아4335', LP_cls)
    cv2.imshow('img', img)
    cv2.waitKey()

    LP_cls = 'P5'
    img = generator.make_LP('서울75마2684', LP_cls)
    cv2.imshow('img', img)
    cv2.waitKey()

    LP_cls = 'P6'
    img = generator.make_LP('12가3456', LP_cls)
    cv2.imshow('img', img)
    cv2.waitKey()
