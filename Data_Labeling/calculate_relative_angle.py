import math
import os

import cv2
import numpy as np

from Data_Labeling.Dataset_Loader.DatasetLoader_WebCrawl import DatasetLoader_WebCrawl
from Data_Labeling.Graphical_Model_Generation.Graphical_Model_Generator_KOR import Graphical_Model_Generator_KOR as GMG
from Utils import imread_uni, add_text_with_background

if __name__ == '__main__':
    prefix = './Dataset_Loader/sample_image_label/PM'
    loader = DatasetLoader_WebCrawl(prefix)

    cv2.namedWindow('img_disp', cv2.WINDOW_NORMAL)
    for l, jpg_path in enumerate(loader.list_jpg):
        img_orig = imread_uni(os.path.join(prefix, jpg_path))
        plate_type, plate_number, xy1, xy2, xy3, xy4, left, top, right, bottom = loader.parse_json(loader.list_json[l])

        if plate_type not in GMG.LP_plate_wh.keys():
            raise NotImplementedError

        # 3D 상의 점
        vw, vh = GMG.LP_plate_wh[plate_type]
        vmax = max(vh, vw)
        vh /= vmax
        vw /= vmax
        canonical_rect = [
            [[-vw / 2], [-vh / 2], [0]], [[vw / 2], [-vh / 2], [0]],
            [[vw / 2], [vh / 2], [0]], [[-vw / 2], [vh / 2], [0]],
        ]

        # virtual camera matrix
        iw, ih = img_orig.shape[1::-1]
        focal_length = max(iw, ih)
        camera_matrix = np.float64([[focal_length, 0, iw / 2],
                                    [0, focal_length, ih / 2],
                                    [0, 0, 1]])
        distortion_matrix = np.zeros((4, 1), dtype=np.float64)

        # quad-box centering
        projected_points_f = np.float32([xy1, xy2, xy3, xy4])
        centering_offset = projected_points_f.mean(axis=0) - np.array([iw, ih]) / 2
        projected_points_f -= centering_offset

        # solvePnP
        pts3d = np.float64(canonical_rect).squeeze(2)
        pts2d = np.float64(projected_points_f)
        success, rot_vec, trans_vec = cv2.solvePnP(pts3d, pts2d, camera_matrix, distortion_matrix, flags=cv2.SOLVEPNP_ITERATIVE)

        if success:
            rmat, jac = cv2.Rodrigues(rot_vec)  # Get rotational matrix
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)  # Get angles
            if math.isnan(any(angles)):
                print('nan')
            else:
                angles_x = -angles[0]
                angles_y = -angles[1]
                angles_z = angles[2]
                print('x:%.2f y:%.2f z:%.2f' % (angles_x, angles_y, angles_z))

        # projection result
        reproj_, jacobian = cv2.projectPoints(pts3d, rot_vec, trans_vec, camera_matrix, distortion_matrix)
        reproj = np.int32(reproj_.squeeze(1) + centering_offset)

        img_disp = img_orig.copy()
        cv2.rectangle(img_disp, (left, top), (right, bottom), (255, 255, 0), 3)  # bounding box
        cv2.polylines(img_disp, [np.int32([xy1, xy2, xy3, xy4])], True, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)  # quadrilateral box
        # cv2.polylines(img_disp, [reproj], True, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        font_size = (right - left) // 5  # magic number
        img_disp = add_text_with_background(img_disp, plate_number, position=(left, top - font_size), font_size=font_size, padding=0)

        cv2.resizeWindow('img_disp', list(map(lambda x: x // 2, img_disp.shape[1::-1])))
        cv2.imshow('img_disp', img_disp)
        cv2.waitKey(0)
