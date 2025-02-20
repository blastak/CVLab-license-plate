import os
import shutil
from pathlib import Path

from natsort import natsorted

from LP_Detection.VIN_LPD.VinLPD import load_model_VinLPD
from LP_Recognition.VIN_OCR.VinOCR import load_model_VinOCR
from Utils import imread_uni, trans_eng2kor_v1p3


def label_validation(input_root, output_root, site='platesmania'):
    r_net = load_model_VinOCR('../../LP_Recognition/VIN_OCR/weight')
    d_net = load_model_VinLPD('../../LP_Detection/VIN_LPD/weight')  # VIN_LPD 사용 준비

    list_dir = natsorted(os.listdir(input_root))
    for _, folder_name in enumerate(list_dir[:]):
        os.makedirs(output_root + '/' + folder_name, exist_ok=True)
        p_in = Path(input_root) / Path(folder_name)
        img_paths = natsorted(p.absolute() for p in p_in.glob('**/*.jpg'))

        print(folder_name)
        for _, img_path in enumerate(img_paths):
            img = imread_uni(img_path)  # 이미지 로드
            if img is None:
                continue

            if site == 'nanoomacar':  # 나눔오토카(청년모터스)
                file_name = img_path.name
                label = file_name.split('_')[2].split('.')[0]  # Plate Number
            else:  # Platesmania
                file_name = img_path.name.replace(' ', '')
                label = file_name.split('_')[1].split('.')[0]  # Plate Number
            img_hash = file_name.split('_')[0][:8]

            d_out = d_net.forward(img)[0]  # VIN_LPD로 검출
            if not d_out:
                continue
            bbs = []
            for bb in d_out:
                crop_resized_img = r_net.keep_ratio_padding(img, bb)
                bbs.append(crop_resized_img)
            r_outs = r_net.forward(bbs)
            for i, r_out in enumerate(r_outs):
                list_char = r_net.check_align(r_out, d_out[i].class_idx + 1)
                list_char_kr = trans_eng2kor_v1p3(list_char)
                plate_number = ''.join(list_char_kr)

                if label == plate_number and len(img_hash) == 8:  # 찾으면 파일 이름 변경, 이동
                    file_name = img_hash + '_' + folder_name + '_' + label + img_path.suffix
                    shutil.move(img_path, os.path.join(output_root, folder_name, file_name))
                    print(f'{label}_저장')
                    break


def platesmania_label():
    input_root = './data'
    output_root = './data2'
    label_validation(input_root, output_root)


def nanoomacar_label():
    input_root = './nanoomacar'
    output_root = './nanoomacar2'
    label_validation(input_root, output_root, 'nanoomacar')


if __name__ == '__main__':
    # platesmania_label()
    nanoomacar_label()  # nanoomacar.py 실행 후 실행
