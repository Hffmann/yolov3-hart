import numpy as np
from mob_utils import KITTI_dataloader

P2 = [7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01, 0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01, 0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]
P2 = np.reshape(P2, (3,4))


def recover_angle(bin_anchor, bin_confidence, bin_num):
    # select anchor from bins
    max_anc = np.argmax(bin_confidence)
    anchors = bin_anchor[max_anc]
    # compute the angle offset
    if anchors[1] > 0:
        angle_offset = np.arccos(anchors[0])
    else:
        angle_offset = -np.arccos(anchors[0])

    # add the angle offset to the center ray of each bin to obtain the local orientation
    wedge = 2 * np.pi / bin_num
    angle = angle_offset + max_anc * wedge

    # angle - 2pi, if exceed 2pi
    angle_l = angle % (2 * np.pi)

    # change to ray back to [-pi, pi]
    angle = angle_l - np.pi / 2
    if angle > np.pi:
        angle -= 2 * np.pi
    angle = round(angle, 2)
    return angle


def compute_orientation(xmax, xmin, alpha):
    x = (xmax + xmin) / 2

    print(" P2: {}".format(P2))

    # compute camera orientation
    u_distance = x - P2[0, 2]
    focal_length = P2[0, 0]
    rot_ray = np.arctan(u_distance / focal_length)

    # global = alpha + ray
    rot_global = alpha + rot_ray

    print(" ALPHA: {}".format(alpha))
    print(" RAY: {}".format(rot_ray))
    print(" ROT_GLOBAL: {}".format(rot_global))

    # local orientation, [0, 2 * pi]
    # rot_local = obj.alpha + np.pi / 2
    rot_local = KITTI_dataloader.get_new_alpha(alpha)

    rot_global = round(rot_global, 2)
    return rot_global, rot_local


def translation_constraints(h, w, l, box_2d, rot_global, rot_local):

    # rotation matrix
    R = np.array([[ np.cos(rot_global), 0,  np.sin(rot_global)],
                  [          0,             1,             0          ],
                  [-np.sin(rot_global), 0,  np.cos(rot_global)]])
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))
    I = np.identity(3)

    xmin_candi, xmax_candi, ymin_candi, ymax_candi = box3d_candidate(h, w, l, rot_local, soft_range=8)

    X  = np.bmat([xmin_candi, xmax_candi,
                  ymin_candi, ymax_candi])
    # X: [x, y, z] in object coordinate
    X = X.reshape(4,3).T

    # construct equation (4, 3)
    for i in range(4):
        matrice = np.bmat([[I, np.matmul(R, X[:,i])], [np.zeros((1,3)), np.ones((1,1))]])
        M = np.matmul(P2, matrice)

        if i % 2 == 0:
            A[i, :] = M[0, 0:3] - box_2d[i] * M[2, 0:3]
            b[i, :] = M[2, 3] * box_2d[i] - M[0, 3]

        else:
            A[i, :] = M[1, 0:3] - box_2d[i] * M[2, 0:3]
            b[i, :] = M[2, 3] * box_2d[i] - M[1, 3]
    # solve x, y, z, using method of least square
    Tran = np.matmul(np.linalg.pinv(A), b)

    tx, ty, tz = [float(np.around(tran, 2)) for tran in Tran]
    return tx, ty, tz


def box3d_candidate(h, w, l, rot_local, soft_range):

    x_corners = [ l, l, l, l, 0, 0, 0, 0 ]
    y_corners = [ h, 0, h, 0, h, 0, h, 0 ]
    z_corners = [ 0, 0, w, w, w, w, 0, 0 ]

    x_corners = [i - l / 2 for i in x_corners]
    y_corners = [i - h for i in y_corners]
    z_corners = [i - w / 2 for i in z_corners]

    corners_3d = np.array([x_corners, y_corners, z_corners])

    '''
    print(" Corners_3d: {}".format(corners_3d))
    '''

    corners_3d = np.transpose(np.array([x_corners, y_corners, z_corners]))

    '''
    print(" Corners_T: {}".format(corners_3d))
    '''

    point1 = corners_3d[0, :]
    point2 = corners_3d[1, :]
    point3 = corners_3d[2, :]
    point4 = corners_3d[3, :]
    point5 = corners_3d[6, :]
    point6 = corners_3d[7, :]
    point7 = corners_3d[4, :]
    point8 = corners_3d[5, :]

    # set up projection relation based on local orientation
    xmin_candi = xmax_candi = ymin_candi = ymax_candi = 0

    if 0 < rot_local < np.pi / 2:
        xmin_candi = point8
        xmax_candi = point2
        ymin_candi = point2
        ymax_candi = point5

    if np.pi / 2 <= rot_local <= np.pi:
        xmin_candi = point6
        xmax_candi = point4
        ymin_candi = point4
        ymax_candi = point1

    if np.pi < rot_local <= 3 / 2 * np.pi:
        xmin_candi = point2
        xmax_candi = point8
        ymin_candi = point8
        ymax_candi = point1

    if 3 * np.pi / 2 <= rot_local <= 2 * np.pi:
        xmin_candi = point4
        xmax_candi = point6
        ymin_candi = point6
        ymax_candi = point5

    # soft constraint
    div = soft_range * np.pi / 180
    if 0 < rot_local < div or 2*np.pi-div < rot_local < 2*np.pi:
        xmin_candi = point8
        xmax_candi = point6
        ymin_candi = point6
        ymax_candi = point5

    if np.pi - div < rot_local < np.pi + div:
        xmin_candi = point2
        xmax_candi = point4
        ymin_candi = point8
        ymax_candi = point1

    return xmin_candi, xmax_candi, ymin_candi, ymax_candi
