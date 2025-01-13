"""
data/ 내의 모든 폴더를 순회하면서
이미지 로딩, 번호판 검출, 인식을 진행
OCR결과_파일해시값_원폴더.jpg 로 파일 이름을 변경하면서 data2/P?/ 로 파일 이동
"""

import os
import shutil
from pathlib import Path

from natsort import natsorted

from Data_Collection.Duplicate_Checker import calc_file_hash
from LP_Detection.VIN_LPD import load_model_VinLPD
from LP_Recognition.VIN_OCR import load_model_VinOCR
from Utils import imread_uni, trans_eng2kor_v1p3

if __name__ == '__main__':
    input_root = './data'
    output_root = './data2'
    for i in range(1, 13):
        os.makedirs(output_root + '/P%d' % i, exist_ok=True)

    r_net = load_model_VinOCR('../../LP_Recognition/VIN_OCR/weight')
    d_net = load_model_VinLPD('../../LP_Detection/VIN_LPD/weight')  # VIN_LPD 사용 준비

    list_dir = natsorted(os.listdir(input_root))
    for _, d in enumerate(list_dir[80:]):
        p_in = Path(input_root) / Path(d)
        img_paths = natsorted(p.absolute() for p in p_in.glob('**/*.jpg'))

        print(d)
        for _, img_path in enumerate(img_paths):
            img = imread_uni(img_path)  # 이미지 로드
            if img is None:
                continue
            d_out = d_net.forward(img)  # VIN_LPD로 검출
            type_ocr_list = []  # 저장 후보 리스트
            for i, bb in enumerate(d_out):
                crop_resized_img = r_net.keep_ratio_padding(img, bb)
                r_out = r_net.forward(crop_resized_img)
                # for b in r_out:
                #     cv2.rectangle(crop_resized_img, (b.x, b.y, b.w, b.h), (255, 255, 0), 1)  # bounding box
                #     font_size = b.w  # magic number
                #     char = bd_eng2kor_v1p3[b.class_str] if not b.class_str.isdigit() else b.class_str
                #     crop_resized_img = add_text_with_background(crop_resized_img, char, position=(b.x, b.y - font_size), font_size=font_size, padding=0).astype(np.uint8)
                list_char_en = r_net.check_align(r_out, bb.class_idx + 1)
                list_char_kr = trans_eng2kor_v1p3(list_char_en)

                total_string = ''.join(list_char_kr)
                cnt_digit = sum([c.isdigit() for c in total_string])

                if bb.class_str in ['P1', 'P2', 'P3', 'P4']:
                    if cnt_digit >= 6:
                        type_ocr_list.append((bb.class_idx, total_string))
                else:
                    if cnt_digit >= 2:
                        type_ocr_list.append((bb.class_idx, total_string))

            if len(type_ocr_list) != 0:
                type_ocr_list.sort(key=lambda x: x[0], reverse=True)

                folder_name = 'P%d' % (type_ocr_list[0][0] + 1)

                plate_number = type_ocr_list[0][1]
                img_md5 = calc_file_hash(img_path)
                cur_dir = d[6:].replace(' ', '')
                file_name = '_'.join([plate_number, img_md5[:8], cur_dir]) + img_path.suffix

                # shutil.copy(img_path, os.path.join(output_root, folder_name, file_name))
                shutil.move(img_path, os.path.join(output_root, folder_name, file_name))
