from math import sin, cos

import cv2
import numpy as np


def find_T_matrix(pts, t_pts):
    A = np.zeros((8, 9))
    for i in range(0, 4):
        xi = pts[:, i];
        xil = t_pts[:, i];
        xi = xi.T

        A[i * 2, 3:6] = -xil[2] * xi
        A[i * 2, 6:] = xil[1] * xi
        A[i * 2 + 1, :3] = xil[2] * xi
        A[i * 2 + 1, 6:] = -xil[0] * xi

    [U, S, V] = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))

    return H


def randomPerturbationPts(pts, alfa=0.02):
    #
    # Adds a random perturbation to a quadrilateral, based on factor alfa
    #

    #
    # Difines the direction of the scaling for each point
    #
    signs = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]])
    sides = []
    pts2 = np.zeros((2, 4))
    for i in range(4):
        sides.append(np.linalg.norm(pts[:, i] - pts[:, (i + 1) % 4]))

    scale = np.array([(sides[0] + sides[2]) / 2 * alfa, (sides[1] + sides[3]) / 2 * alfa])
    for i in range(4):
        pts2[:, i] = pts[:, i] + np.random.rand(2) * signs[:, i] * scale
    return pts2


def CropWarpImage(img, pts, outsize):
    t_ptsh = getRectPts(0, 0, outsize[0], outsize[1])
    ptsh = np.concatenate((pts, np.ones((1, 4))))
    H = find_T_matrix(ptsh, t_ptsh)
    Ilp = cv2.warpPerspective(img, H, outsize, flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, borderValue=.0)
    return Ilp


def getRectPts(tlx, tly, brx, bry):
    return np.matrix([[tlx, brx, brx, tlx], [tly, tly, bry, bry], [1., 1., 1., 1.]], dtype=float)


def perspective_transform(wh, angles=np.array([0., 0., 0.]), zcop=1000., dpp=1000.):
    rads = np.deg2rad(angles)

    a = rads[0];
    Rx = np.matrix([[1, 0, 0], [0, cos(a), sin(a)], [0, -sin(a), cos(a)]])
    a = rads[1];
    Ry = np.matrix([[cos(a), 0, -sin(a)], [0, 1, 0], [sin(a), 0, cos(a)]])
    a = rads[2];
    Rz = np.matrix([[cos(a), sin(a), 0], [-sin(a), cos(a), 0], [0, 0, 1]])

    R = Rx * Ry * Rz;

    (w, h) = tuple(wh)
    xyz = np.matrix([[0, 0, w, w], [0, h, 0, h], [0, 0, 0, 0]])
    hxy = np.matrix([[0, 0, w, w], [0, h, 0, h], [1, 1, 1, 1]])

    xyz = xyz - np.matrix([[w], [h], [0]]) / 2.  ###### hrkim: centering
    xyz = R * xyz

    xyz = xyz - np.matrix([[0], [0], [zcop]])
    hxyz = np.concatenate([xyz, np.ones((1, 4))])

    P = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1. / dpp, 0]])
    _hxy = P * hxyz
    _hxy = _hxy / _hxy[2, :]
    _hxy = _hxy + np.matrix([[w], [h], [0]]) / 2.  ###### hrkim: de-centering

    return find_T_matrix(hxy, _hxy)
