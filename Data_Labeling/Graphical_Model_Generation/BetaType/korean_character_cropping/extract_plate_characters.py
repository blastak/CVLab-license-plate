import os

import cv2
import numpy as np

from LP_Detection import BBox
from LP_Detection.VIN_LPD import load_model_VinLPD
from LP_Recognition.VIN_OCR import load_model_VinOCR
from Utils import imread_uni, imwrite_uni, trans_eng2kor_v1p3

# P5
h_pt = [14, 54, 66, 156]
h_pt2 = [50, 126]
v_pt1 = [90, 125, 134, 169, 178, 208, 215, 245]
v_pt2 = [22, 82, 105, 150, 161, 206, 217, 262, 273, 318]
char_n = [(14, 50, 90, 125), (14, 50, 134, 169), (14, 54, 178, 208), (14, 54, 215, 245), (66, 126, 22, 82),
          (66, 156, 105, 150), (66, 156, 161, 206), (66, 156, 217, 262), (66, 156, 273, 318)]


# P6
# h_pt = [13, 50, 64, 156]
# v_pt1 = [90, 135, 145, 190, 200, 245]
# v_pt2 = [15, 80, 95, 160, 175, 240, 255, 320]
# char_n = [(13, 50, 90, 135), (13, 50, 145, 190), (13, 50, 200, 245), (64, 156, 15, 80), (64, 156, 95, 160),
#           (64, 156, 175, 240), (64, 156, 255, 320)]


def transform_to_plane(image, src_points):
    height, width = image.shape[:2]
    src_points = np.float32(src_points)
    dst_points = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    result = cv2.warpPerspective(image, matrix, (width, height))
    return result


def resize_image(image, b):
    BBox = np.float32([(b.x, b.y), (b.x + b.w, b.y), (b.x + b.w, b.y + b.h), (b.x, b.y + b.h)])
    x_min = max(int(np.min(BBox[:, 0])) - 30, 0)
    x_max = min(int(np.max(BBox[:, 0])) + 30, image.shape[1] - 1)
    y_min = max(int(np.min(BBox[:, 1])) - 30, 0)
    y_max = min(int(np.max(BBox[:, 1])) + 30, image.shape[0] - 1)
    cropped_image = img[y_min:y_max, x_min:x_max]
    return cv2.resize(cropped_image, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)


def edit_coordinate(image, label, click_points, img_path, save_path):
    cv2.setMouseCallback("Resized Image", mouse_callback, param={'click_points': click_points})
    cv2.waitKey(0)  # 4꼭지점 클릭 후 엔터
    img_copy = image.copy()
    for point in click_points:
        cv2.circle(img_copy, point, 5, (0, 0, 255), -1)
    cv2.imshow("Resized Image", img_copy)
    if len(click_points) == 4:
        transformed_image = transform_to_plane(image, click_points)
        draw_lines(transformed_image)
    i = 0
    while True:
        if len(click_points) != 4:
            print('-------exit-------')
            break
        key = cv2.waitKeyEx()
        if key == ord('1'):  # 왼쪽 위
            i = 0
        elif key == ord('2'):  # 오른쪽 위
            i = 1
        elif key == ord('3'):  # 오른쪽 아래
            i = 2
        elif key == ord('4'):  # 왼쪽 아래
            i = 3
        elif key == 2424832:  # 왼쪽 화살표 키
            click_points[i] = (click_points[i][0] + 1, click_points[i][1])
        elif key == 2490368:  # 위쪽 화살표 키
            click_points[i] = (click_points[i][0], click_points[i][1] + 1)
        elif key == 2555904:  # 오른쪽 화살표 키
            click_points[i] = (click_points[i][0] - 1, click_points[i][1])
        elif key == 2621440:  # 아래쪽 화살표 키
            click_points[i] = (click_points[i][0], click_points[i][1] - 1)
        elif key == 13:  # 엔터 저장
            imwrite_uni(os.path.join(save_path, f'{label}.jpg'), transform_to_plane(image, click_points))
            # save_img(transformed_image,label,save_path)
            print('-------save-------')
            print(click_points)
            break
        elif key == ord('s'):  # transform 이미지만 저장
            imwrite_uni(os.path.join(save_path, f'{label}.jpg'), transform_to_plane(image, click_points))
            print('-------save-------')
            print(click_points)
            break
        elif key == ord('e'):  # edit 모드 종료
            print(click_points)
            print('-------exit-------')
            break
        transformed_image = transform_to_plane(resized_image, click_points)
        # draw_point(image, click_points)
        img_copy = image.copy()
        draw_points(img_copy, click_points)
        draw_lines(transformed_image)


def mouse_callback(event, x, y, flags, param):
    click_points = param['click_points']
    if len(click_points) == 4:
        return
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼 클릭 시
        click_points.append((x, y))
        print(f"Clicked: ({x}, {y})")


def draw_points(img, points):
    for point in points:
        cv2.circle(img, point, 5, (0, 0, 255), -1)
    cv2.imshow("Resized Image", img)


def draw_lines(image):
    for h in h_pt:  # P5
        h = int(h * image.shape[0] / 170)
        cv2.line(image, (0, h), (int(335 * image.shape[1] / 335), h), (0, 0, 0), 1)
    cv2.line(image, (0, int(50 * image.shape[0] / 170)), (int(169 * image.shape[1] / 335), int(50 * image.shape[0] / 170)), (0, 0, 0), 1)
    cv2.line(image, (0, int(126 * image.shape[0] / 170)), (int(82 * image.shape[1] / 335), int(126 * image.shape[0] / 170)), (0, 0, 0), 1)
    for v in v_pt1:
        v = int(v * image.shape[1] / 335)
        cv2.line(image, (v, 0), (v, int(54 * image.shape[0] / 170)), (0, 0, 0), 1)
    for v in v_pt2:
        v = int(v * image.shape[1] / 335)
        cv2.line(image, (v, int(66 * image.shape[0] / 170)), (v, int(170 * image.shape[0] / 170)), (0, 0, 0), 1)
    # for h in h_pt:    # P6
    #     h = int(h * image.shape[0] / 170)
    #     cv2.line(image, (0, h), (int(335*image.shape[1]/335), h), (0, 0, 0), 1)
    # for v in v_pt1:
    #     v = int(v * image.shape[1] / 335)
    #     cv2.line(image, (v, 0), (v, int(50*image.shape[0]/170)), (0, 0, 0), 1)
    # for v in v_pt2:
    #     v = int(v * image.shape[1] / 335)
    #     cv2.line(image, (v, int(64*image.shape[0]/170)), (v, int(170*image.shape[0]/170)), (0, 0, 0), 1)
    cv2.imshow("Transformed_image", image)


def save_img(T_image, label, save_path):
    for i, (a, b, c, d) in enumerate(char_n):
        if i == 0 or i == 1:
            continue
        # if i <2:  # P6
        elif i == 2 or i == 3:  # P5
            save_folder = os.path.join(save_path, f'_{label[i]}')
        else:
            save_folder = os.path.join(save_path, label[i])
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        exist = os.listdir(save_folder)
        exist_n = []
        for file in exist:
            if file.endswith('.jpg'):
                try:
                    number = int(file.split('.')[0])  # 파일명에서 숫자만 추출
                    exist_n.append(number)
                except ValueError:
                    continue
        if exist_n:
            next_number = max(exist_n) + 1
        else:
            next_number = 0

        cropped_image = T_image[a:b, c:d]
        imwrite_uni(os.path.join(save_folder, f'{next_number}.jpg'), cropped_image)
        next_number += 1


if __name__ == "__main__":
    prefix_path = r"D:\Dataset\LicensePlate\for_p5p6\extract\HR"
    move_path = r"D:\Dataset\LicensePlate\for_p5p6\extract\move_to"
    save_path = r"D:\Dataset\LicensePlate\for_p5p6\extract\P5_pre"
    img_paths = [a for a in os.listdir(prefix_path) if a.endswith('.jpg')]

    r_net = load_model_VinOCR('../../../../LP_Recognition/VIN_OCR/weight')
    d_net = load_model_VinLPD('../../../../LP_Detection/VIN_LPD/weight')  # VIN_LPD 사용 준비

    for _, img_path in enumerate(img_paths):
        img = imread_uni(os.path.join(prefix_path, img_path))  # 이미지 로드
        i_h, i_w = img.shape[:2]
        boxes = []
        d_out = d_net.forward(img)[0]
        for _, d in enumerate(d_out):
            bb_vinlpd = BBox(d.x, d.y, d.w, d.h, d.class_str, d.class_idx)
            boxes.append(bb_vinlpd)

        process = 'continue'
        for _, bb in enumerate(boxes):
            crop_resized_img = r_net.keep_ratio_padding(img, bb)  # 글자 인식
            r_out = r_net.forward(crop_resized_img)
            bb.class_idx = 4
            list_char = r_net.check_align(r_out, bb.class_idx + 1)
            list_char_kr = trans_eng2kor_v1p3(list_char)
            label = ''.join(list_char_kr)
            print(label)
            cv2.imshow("Image", img)  # 원본 이미지
            resized_image = resize_image(img, bb)
            cv2.imshow("Resized Image", resized_image)  # 번호판 ROI
            key = cv2.waitKey()
            if key == 27:  # 'esc' 프로세스 종료
                process = 'exit'
                break
            elif key == ord('p'):  # 'p' 키 패스
                continue
            elif key == ord('e'):  # 'e' 수정
                print('-----edit mode-----')
                click_points = []
                edit_coordinate(resized_image, label, click_points, img_path, save_path)
                # os.rename(os.path.join(prefix_path, img_path),os.path.join(move_path, img_path))
            elif key == ord('m'):  # 'm' 이동
                os.rename(os.path.join(prefix_path, img_path), os.path.join(move_path, img_path))
            elif key == ord('d'):  # 'd' 파일 삭제
                os.remove(os.path.join(prefix_path, img_path))
        if process == 'exit':  # 'esc' 프로세스 종료
            break
