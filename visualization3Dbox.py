import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
from PIL import Image

from mob_utils.read_dir import ReadDir
from config import config as cfg
from mob_utils.correspondece_constraint import *

import cv2

'''
def compute_birdviewbox(line, shape, scale):
    npline = [np.float64(line[i]) for i in range(1, len(line))]
    h = npline[7] * scale
    w = npline[8] * scale
    l = npline[9] * scale
    x = npline[10] * scale
    y = npline[11] * scale
    z = npline[12] * scale
    rot_y = npline[13]

    R = np.array([[-np.cos(rot_y), np.sin(rot_y)],
                  [np.sin(rot_y), np.cos(rot_y)]])
    t = np.array([x, z]).reshape(1, 2).T

    x_corners = [0, l, l, 0]  # -l/2
    z_corners = [w, w, 0, 0]  # -w/2


    x_corners += -w / 2
    z_corners += -l / 2

    # bounding box in object coordinate
    corners_2D = np.array([x_corners, z_corners])
    # rotate
    corners_2D = R.dot(corners_2D)
    # translation
    corners_2D = t - corners_2D
    # in camera coordinate
    corners_2D[0] += int(shape/2)
    corners_2D = (corners_2D).astype(np.int16)
    corners_2D = corners_2D.T

    return np.vstack((corners_2D, corners_2D[0,:]))

def draw_birdeyes(ax2, line_gt, line_p, shape):
    # shape = 900
    scale = 15

    pred_corners_2d = compute_birdviewbox(line_p, shape, scale)
    gt_corners_2d = compute_birdviewbox(line_gt, shape, scale)

    codes = [Path.LINETO] * gt_corners_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(gt_corners_2d, codes)
    p = patches.PathPatch(pth, fill=False, color='orange', label='ground truth')
    ax2.add_patch(p)

    codes = [Path.LINETO] * pred_corners_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(pred_corners_2d, codes)
    p = patches.PathPatch(pth, fill=False, color='green', label='prediction')
    ax2.add_patch(p)
'''


def compute_3Dbox(P2, box_2d, rot_global, h, w, l, tx, ty, tz):

    xmin, ymin, xmax, ymax = box_2d

    '''
    obj = detectionInfo(line)
    # Draw 2D Bounding Box
    xmin = int(obj.xmin)
    xmax = int(obj.xmax)
    ymin = int(obj.ymin)
    ymax = int(obj.ymax)
    # width = xmax - xmin
    # height = ymax - ymin
    # box_2d = patches.Rectangle((xmin, ymin), width, height, fill=False, color='red', linewidth='3')
    # ax.add_patch(box_2d)
    '''
    # Draw 3D Bounding Box

    R = np.array([[np.cos(rot_global), 0, np.sin(rot_global)],
                  [0, 1, 0],
                  [-np.sin(rot_global), 0, np.cos(rot_global)]])

    x_corners = [0, l, l, l, l, 0, 0, 0]  # -l/2
    y_corners = [0, 0, h, h, 0, 0, h, h]  # -h
    z_corners = [0, 0, 0, w, w, w, w, 0]  # -w/2

    x_corners = [i - l / 2 for i in x_corners]
    y_corners = [i - h for i in y_corners]
    z_corners = [i - w / 2 for i in z_corners]

    corners_3D = np.array([x_corners, y_corners, z_corners])
    corners_3D = R.dot(corners_3D)
    corners_3D += np.array([tx, ty, tz]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    corners_2D = P2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[:2]

    return corners_2D

def draw_3Dbox(frame, color, box_2d, rot_global, h, w, l, tx, ty, tz):

    corners_2D = compute_3Dbox(P2, box_2d, rot_global, h, w, l, tx, ty, tz)

    # draw all lines through path
    # https://matplotlib.org/users/path_tutorial.html
    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
    bb3d_on_2d_lines_verts = corners_2D[:, bb3d_lines_verts_idx]
    verts = bb3d_on_2d_lines_verts.T

    '''
    print(" corners_2D: {} ".format(corners_2D))
    print(" bb3d_lines_verts_idx: {} ".format(bb3d_lines_verts_idx))
    print(" bb3d_on_2d_lines: {} ".format(bb3d_on_2d_lines_verts))

    print(" Verts: {} ".format(verts))
    '''

    for i in range(len(verts) - 1):
        cv2.line(frame, (int(verts[i][0]), int(verts[i][1])), (int(verts[i+1][0]), int(verts[i+1][1])), color, 2)
