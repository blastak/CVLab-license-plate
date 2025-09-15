import math
import os

import cv2
import numpy as np

from Data_Labeling.Dataset_Loader.DatasetLoader_WebCrawl import DatasetLoader_WebCrawl
from Data_Labeling.Graphical_Model_Generation.Graphical_Model_Generator_KOR import Graphical_Model_Generator_KOR as GMG
from Utils import imread_uni, add_text_with_background


def calc_relative_angle(xy1, xy2, xy3, xy4, plate_type, image_width, image_height):
    """
    이미지 내 번호판의 상대적인 각도 계산

    이 함수는 이미지 내 번호판의 네 꼭짓점 좌표, 번호판 종류, 이미지 크기를 입력받아
    3차원 공간에서 번호판의 회전 각도를 추정합니다.

    Args:
        xy1 (tuple): 번호판 왼쪽 상단 꼭짓점 좌표
        xy2 (tuple): 번호판 오른쪽 상단 꼭짓점 좌표
        xy3 (tuple): 번호판 오른쪽 하단 꼭짓점 좌표
        xy4 (tuple): 번호판 왼쪽 하단 꼭짓점 좌표
        plate_type (str): 번호판 종류
        image_width (int): 이미지 너비
        image_height (int): 이미지 높이

    Returns:
        list: 번호판의 상대적인 회전 각도 (x, y, z)를 요소로 하는 리스트 (단위: 도)
    """

    if plate_type not in GMG.plate_wh.keys():
        raise NotImplementedError

    # 3D 상의 점
    vw, vh = GMG.plate_wh[plate_type]
    vmax = max(vh, vw)
    vh /= vmax
    vw /= vmax
    canonical_rect = [
        [[-vw / 2], [-vh / 2], [0]], [[vw / 2], [-vh / 2], [0]],
        [[vw / 2], [vh / 2], [0]], [[-vw / 2], [vh / 2], [0]],
    ]

    # virtual camera matrix
    focal_length = max(image_width, image_height)
    camera_matrix = np.float64([[focal_length, 0, image_width / 2],
                                [0, focal_length, image_height / 2],
                                [0, 0, 1]])
    distortion_matrix = np.zeros((4, 1), dtype=np.float64)

    # quad-box centering
    projected_points_f = np.float32([xy1, xy2, xy3, xy4])
    centering_offset = projected_points_f.mean(axis=0) - np.array([image_width, image_height]) / 2
    projected_points_f -= centering_offset  # centering을 하지 않으면 translation 값 때문에 rotation의 해석이 어렵게 되어버린다.

    # solvePnP
    pts3d = np.float64(canonical_rect).squeeze(2)
    pts2d = np.float64(projected_points_f)
    success, rot_vec, trans_vec = cv2.solvePnP(pts3d, pts2d, camera_matrix, distortion_matrix, flags=cv2.SOLVEPNP_ITERATIVE)

    # # reprojection test
    # reproj_, jacobian = cv2.projectPoints(pts3d, rot_vec, trans_vec, camera_matrix, distortion_matrix)
    # reproj = np.int32(reproj_.squeeze(1) + centering_offset)

    angles_in_deg = [0, 0, 0]
    if success:
        rmat, jac = cv2.Rodrigues(rot_vec)  # Get rotational matrix
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)  # Get angles
        if math.isnan(any(angles)):
            angles_in_deg = [0, 0, 0]
        else:
            angles_in_deg = [-angles[0], -angles[1], angles[2]]
    return angles_in_deg


if __name__ == '__main__':
    # prefix = './Dataset_Loader/sample_image_label/PM'
    # loader = DatasetLoader_WebCrawl(prefix)
    #
    # cv2.namedWindow('img_disp', cv2.WINDOW_NORMAL)
    # for l, jpg_path in enumerate(loader.list_jpg):
    #     img_orig = imread_uni(os.path.join(prefix, jpg_path))
    #     plate_type, plate_number, xy1, xy2, xy3, xy4, left, top, right, bottom = loader.parse_json(loader.list_json[l])
    #
    #     iw, ih = img_orig.shape[1::-1]
    #     angle_xyz = calc_relative_angle(xy1, xy2, xy3, xy4, plate_type, iw, ih)
    #     print('x:%-10.2f y:%-10.2f z:%-10.2f' % tuple(angle_xyz))
    #
    #     img_disp = img_orig.copy()
    #     cv2.rectangle(img_disp, (int(left), int(top)), (int(right), int(bottom)), (255, 255, 0), 3)  # bounding box
    #     cv2.polylines(img_disp, [np.int32([xy1, xy2, xy3, xy4])], True, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)  # quadrilateral box
    #     font_size = (right - left) // 5  # magic number
    #     img_disp = add_text_with_background(img_disp, plate_number, position=(left, top - font_size), font_size=font_size, padding=0)
    #
    #     cv2.resizeWindow('img_disp', list(map(lambda x: x // 2, img_disp.shape[1::-1])))
    #     cv2.imshow('img_disp', img_disp)
    #     cv2.waitKey(0)

    from pathlib import Path
    import json
    base_dir = '/workspace/DB/01_LicensePlate/55_WebPlatemania_jpg_json_20250407'
    for folder in Path(base_dir).glob('GoodMatches_*'):
        print(f"\n처리 중: {folder.name}")
        loader = DatasetLoader_WebCrawl(str(folder))

        for i, json_file in enumerate(loader.list_json):
            json_path = folder / json_file
            jpg_path = folder / loader.list_jpg[i]

            # 기존 파싱 로직
            img = imread_uni(str(jpg_path))
            plate_type, plate_number, xy1, xy2, xy3, xy4, _, _, _, _ = \
                loader.parse_json(json_file)

            # 각도 계산
            ih, iw = img.shape[:2]
            angle_xyz = calc_relative_angle(xy1, xy2, xy3, xy4, plate_type, iw, ih)

            # JSON 업데이트
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            data['flags']['angle'] = {
                'x': round(angle_xyz[0], 2),
                'y': round(angle_xyz[1], 2),
                'z': round(angle_xyz[2], 2)
            }

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            if (i + 1) % 100 == 0:
                print(f"  - {i + 1}개 처리 완료")
