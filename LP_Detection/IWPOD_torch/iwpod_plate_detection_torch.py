import cv2
import numpy as np
import torch

from LP_Detection.IWPOD_tf.iwpod_plate_detection_Min import cal_BB
from Utils import imread_uni
from src.detect import detect_lp_width
from src.src.model import IWPODNet
from src.src.utils import im2single


def torch_load_model(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mymodel = IWPODNet()
    mymodel.load_state_dict(torch.load(path)['model_state_dict'])
    mymodel.to(device)
    return mymodel


def find_lp_corner(img_orig, iwpod_net):
    lp_threshold = 0.35  # default 0.35
    ocr_input_size = [80, 240]  # desired LP size (width x height)

    Ivehicle = img_orig
    iwh = np.array(Ivehicle.shape[1::-1], dtype=float).reshape((2, 1))

    ASPECTRATIO = 1
    WPODResolution = 480  # larger if full image is used directly
    lp_output_resolution = tuple(ocr_input_size[::-1])
    Llp, LlpImgs, _ = detect_lp_width(iwpod_net, im2single(Ivehicle), WPODResolution * ASPECTRATIO, 2 ** 4,
                                      lp_output_resolution, lp_threshold)

    xys2_list = []
    for j, img in enumerate(LlpImgs):
        pts = Llp[j].pts * iwh
        xys2 = np.transpose(pts.astype(np.int32))
        xys2_list.append(xys2.tolist())
    return xys2_list


if __name__ == '__main__':
    mymodel = torch_load_model('./src/weights/iwpodnet_retrained_epoch10000.pth')

    # img_path = "../sample_image/seoulmp4_001036359jpg.jpg"
    img_path = "../sample_image/14266136_P1-2_01루4576.jpg"
    img = imread_uni(img_path)
    x = find_lp_corner(img, mymodel)
    y = cal_BB(x)
    print(x)
    print(y)

    img_bb_qb = img.copy()
    for i, b in enumerate(y):
        cv2.rectangle(img_bb_qb, (b.x, b.y, b.w, b.h), (255, 255, 0), 3)  # bounding box
        cv2.polylines(img_bb_qb, [np.int32(x[i])], True, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)  # quadrilateral box
    # cv2.namedWindow('img_bb_qb', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('img_bb_qb', tuple(map(lambda x: int(x * 0.9), (1920, 1080))))
    cv2.imshow('img_bb_qb', img_bb_qb)
    cv2.waitKey()

    # negative test
    neg_img = np.zeros_like(img)
    x = find_lp_corner(neg_img, mymodel)
    y = cal_BB(x)
    print(x)
    print(y)
