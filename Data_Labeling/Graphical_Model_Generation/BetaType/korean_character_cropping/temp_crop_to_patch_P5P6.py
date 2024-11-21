# 한국번호판의 한 글자 crop 한 걸로 규정사이즈로 만들기 (patch로 만들기)

import os

import cv2
import numpy as np

if __name__ == "__main__":
    plate = 'P5'  # 번호판 종류
    save_path = f'./03_patch/{plate}'
    os.makedirs(save_path, exist_ok=True)
    ############# 1
    i = 1
    img_crop_1 = cv2.imread(f'./02_crop/{plate}/_1/{i}_1.png', cv2.IMREAD_UNCHANGED)
    img_crop = cv2.imread(f'./02_crop/{plate}/{i}.png', cv2.IMREAD_UNCHANGED)
    img_bg_1 = np.full([177, 132, 4], 0, dtype=np.uint8)  # _1 숫자 패치 사이즈
    img_bg = np.full([185, 92, 4], 0, dtype=np.uint8)  # 숫자 패치 사이즈
    offset_x_1 = 50
    offset_x = 37
    offset_y = 0
    img_bg_1[offset_y:offset_y + img_crop_1.shape[0], offset_x_1:offset_x_1 + img_crop_1.shape[1]] = img_crop_1
    img_bg[offset_y:offset_y + img_crop.shape[0], offset_x:offset_x + img_crop.shape[1]] = img_crop
    cv2.imwrite(f'{save_path}/{i}_1.png', img_bg_1)
    cv2.imwrite(f'{save_path}/{i}.png', img_bg)


    ############# 0~9
    # for i in range(10):
    #     img_crop_1 = cv2.imread(f'./02_crop/{plate}/_1/{i}_1.png', cv2.IMREAD_UNCHANGED)
    #     img_crop = cv2.imread(f'./02_crop/{plate}/{i}.png', cv2.IMREAD_UNCHANGED)
    #     img_bg_1 = np.full([37, 45, 4], 0, dtype=np.uint8)  # _1 숫자 패치 사이즈
    #     img_bg = np.full([92, 65, 4], 0, dtype=np.uint8)  # 숫자 패치 사이즈
    #     if i == 1:  # 1은 알파 채널 추가
    #         img_crop_1 = cv2.merge(
    #             [img_crop_1, np.ones((img_crop_1.shape[0], img_crop_1.shape[1]), dtype=np.uint8) * 255])
    #         img_crop = cv2.merge([img_crop, np.ones((img_crop.shape[0], img_crop.shape[1]), dtype=np.uint8) * 255])
    #     offset_x_1 = (img_bg_1.shape[1] - img_crop_1.shape[1]) // 2
    #     offset_x = (img_bg.shape[1] - img_crop.shape[1]) // 2
    #     # if i==1:
    #     #     offset_x = 17.5  # 1_1의 X 오프셋
    #     #     # offset_x = 26  # 1의 X 오프셋
    #     offset_y = 0
    #     img_bg_1[offset_y:offset_y + img_crop_1.shape[0], offset_x_1:offset_x_1 + img_crop_1.shape[1]] = img_crop_1
    #     img_bg[offset_y:offset_y + img_crop.shape[0], offset_x:offset_x + img_crop.shape[1]] = img_crop
    #
    #     cv2.imwrite(f'{save_path}/{i}_1.png', img_bg_1)
    #     cv2.imwrite(f'{save_path}/{i}.png', img_bg)
    # #
    # #
    # # ############ 가거고구 나너노누 다더도두 라러로루 마머모무 버보부 서소수 어오우 저조주 하허호 육해공국합
    # list_all = os.listdir(f'./02_crop/{plate}/')[10:-1]
    # offset_x = 0  # X 오프셋
    # offset_y = 0  # Y 오프셋
    # img_sh = [37, 45, 4]
    # for f in list_all:
    #     img_crop = cv2.imread(f'./02_crop/{plate}/{f}', cv2.IMREAD_UNCHANGED)
    #
    #     img_bg = np.full(img_sh, 0, dtype=np.uint8)
    #     img_bg[offset_y:offset_y + img_crop.shape[0], offset_x:offset_x + img_crop.shape[1]] = img_crop
    #     cv2.imwrite(f'{save_path}/{f}', img_bg)
