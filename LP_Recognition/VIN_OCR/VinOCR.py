import os

import cv2
import numpy as np

from Data_Labeling.Dataset_Loader.DatasetLoader_ParkingView import DatasetLoader_ParkingView
from LP_Detection.Bases import OcvYoloBase, BBox
from LP_Detection.VIN_LPD import load_model_VinLPD
from Utils import imread_uni, bd_eng2kor_v1p3, add_text_with_background, trans_eng2kor_v1p3

inpWidth = 256
inpHeight = 224


class VinOCR(OcvYoloBase):
    def __init__(self, _model_path, _weight_path, _classes_path):
        super().__init__(_model_path, _weight_path, _classes_path, _conf_thresh=0.03, _iou_thresh=0.3, _in_w=inpWidth, _in_h=inpHeight)

    def keep_ratio_padding(self, img, b, target_width=inpWidth, target_height=inpHeight):
        if b.x < 0:
            b.w += b.x
            b.x = 0
        if b.y < 0:
            b.h += b.y
            b.y = 0

        # Step 1: Bounding box 영역을 잘라냄
        cropped_img = img[b.y:b.y + b.h, b.x:b.x + b.w, :]

        # Step 2: aspect ratio를 유지하면서 target 크기에 맞춰 resize
        aspect_ratio_target = target_width / target_height
        aspect_ratio_cropped = cropped_img.shape[1] / cropped_img.shape[0]

        if aspect_ratio_cropped > aspect_ratio_target:
            # 폭 기준으로 조정
            new_width = target_width
            new_height = int(cropped_img.shape[0] * target_width / cropped_img.shape[1])
        else:
            # 높이 기준으로 조정
            new_height = target_height
            new_width = int(cropped_img.shape[1] * target_height / cropped_img.shape[0])

        resized_img = cv2.resize(cropped_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Step 3: 패딩 추가
        padded_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        offset_x = (target_width - new_width) // 2
        offset_y = (target_height - new_height) // 2

        padded_img[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized_img

        return padded_img

    def check_align(self, _boxes, _p_type):
        self.check_border(_boxes)  # 오검출 제거
        list_char = self.align_char_box(_boxes, _p_type)  # 결과 정렬
        return list_char

    def check_border(self, _boxes):
        if len(_boxes) < 4:
            return

        ### qsort(&yolos[0], yolos.size(), sizeof(yolos[0]), qCompH); // Height순 정렬
        ### int top3Height = yolos[yolos.size() - 3].obRect.height;
        top3Height = sorted(_boxes, key=lambda b: b.h)[-3].h

        ### qsort(&yolos[0], yolos.size(), sizeof(yolos[0]), qCompW); // Width순 정렬
        ### int top3Width = yolos[yolos.size() - 3].obRect.width;
        top3Width = sorted(_boxes, key=lambda b: b.w)[-3].w

        ### qsort(&yolos[0], yolos.size(), sizeof(yolos[0]), qCompX); // X축 정렬
        _boxes = sorted(_boxes, key=lambda b: b.x)

        leftOneH = _boxes[0].h
        leftOneW = _boxes[0].w
        rightOneH = _boxes[-1].h
        rightOneW = _boxes[-1].w

        if (0.8 <= rightOneH / top3Height <= 1.25) is False and len(_boxes[-1].class_str) == 1:
            del (_boxes[-1])
        elif rightOneW < top3Width / 2.7:
            del (_boxes[-1])

        if (0.8 <= leftOneH / top3Height <= 1.25) is False and len(_boxes[0].class_str) == 1:
            del (_boxes[0])
        elif leftOneW < top3Width / 2.7:
            del (_boxes[0])

    def align_char_box(self, _boxes, _p_type):
        retval = []
        if _p_type == 1 or _p_type == 2 or _p_type == 4 or _p_type == 10 or _p_type == 12:
            if len(_boxes) <= 2: return retval
            _boxes = sorted(_boxes, key=lambda b: b.x)

            if _p_type == 4 or _p_type == 10 or _p_type == 12:  # 지역문자 특수 처리
                while len(_boxes) >= 2:
                    if len(_boxes[0].class_str) == 1 and len(_boxes[1].class_str) != 1:
                        if _boxes[1].x <= inpWidth * 0.2:
                            del (_boxes[0])
                            continue
                    else:
                        if len(_boxes[1].class_str) == 1:  # 제일 왼쪽에서 두번째가 글자가 아닐경우, 한글자만으로 추측한다 (나중에 추가 코드 추가)
                            retval.append(_boxes[0].class_str)
                        else:
                            if _boxes[1].y < _boxes[0].y:
                                _boxes[0], _boxes[1] = _boxes[1], _boxes[0]
                            retval.append(_boxes[0].class_str)
                    break
            else:
                retval.append(_boxes[0].class_str)

            for b in _boxes[1:]:
                retval.append(b.class_str)

        else:  # P3 등 두줄
            hmin = 99999
            hmax = -1
            ymax = -1
            for b in _boxes:
                if hmin > b.y:
                    hmin = b.y
                if hmax < b.y + b.h:
                    if b.conf < 0.6: continue
                    hmax = b.y + b.h
                    ymax = b.y

            top_boxes = [b for b in _boxes if ymax > b.y + b.h]
            for t in top_boxes:
                _boxes.remove(t)

            if len(top_boxes) >= 2:
                top_boxes = sorted(top_boxes, key=lambda b: b.x)

            if len(_boxes) >= 2:
                _boxes = sorted(_boxes, key=lambda b: b.x)

            for t in top_boxes:
                retval.append(t.class_str)
            for b in _boxes:
                retval.append(b.class_str)

        return retval


def load_model_VinOCR(path_base):
    if path_base[-1] != '/':
        path_base += '/'
    d_net = VinOCR(path_base + 'yolov3-rn83.cfg', path_base + 'yolov3-rn83.weights', path_base + 'yolov3-rn83.names')
    return d_net


if __name__ == '__main__':
    r_net = load_model_VinOCR('./weight')

    use_detector = True

    if use_detector:
        img = imread_uni('../../LP_Detection/sample_image/seoulmp4_001036359jpg.jpg')
        d_net = load_model_VinLPD('../../LP_Detection/VIN_LPD/weight/')
        d_out = d_net.forward(img)[0]  # from Detector
    else:
        prefix = '../../Data_Labeling/Dataset_Loader/sample_image_label/파클'
        # prefix = '../../Data_Labeling/Dataset_Loader/sample_image_label/용산'
        # prefix = '../../Data_Labeling/Dataset_Loader/sample_image_label/충남'
        loader = DatasetLoader_ParkingView(prefix)
        jpg_paths = [a for a in os.listdir(prefix) if a.endswith('.jpg')]
        jpg_path = jpg_paths[0]  # 테스트를 위해 하나만 선택
        img = imread_uni(os.path.join(prefix, jpg_path))
        plate_type, plate_number, left, top, right, bottom = loader.parse_info(jpg_path[:-4] + '.xml')  # from XML
        d_out = [BBox(left, top, right - left, bottom - top, plate_number, int(plate_type[1:2]) - 1, 1.0)]

    bbs = []
    for bb in d_out:
        crop_resized_img = r_net.keep_ratio_padding(img, bb)
        bbs.append(crop_resized_img)
    r_outs = r_net.forward(bbs)
    for i, r_out in enumerate(r_outs):
        print(f'[P{d_out[i].class_idx + 1}]', end=' ')
        for cb in r_out:
            cv2.rectangle(bbs[i], (cb.x, cb.y, cb.w, cb.h), (255, 255, 0), 1)  # bounding box
            font_size = cb.w  # magic number
            char = bd_eng2kor_v1p3[cb.class_str] if not cb.class_str.isdigit() else cb.class_str
            bbs[i] = add_text_with_background(bbs[i], char, position=(cb.x, cb.y - font_size), font_size=font_size, padding=0).astype(np.uint8)
            print(char, end='')
        list_char = r_net.check_align(r_out, d_out[i].class_idx + 1)
        list_char_kr = trans_eng2kor_v1p3(list_char)
        print(' -->', ''.join(list_char_kr))
        cv2.imshow(''.join(list_char), bbs[i])
        cv2.waitKey()
