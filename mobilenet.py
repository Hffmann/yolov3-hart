import os
import numpy as np
import cv2
import argparse
from mob_utils.read_dir import ReadDir
from mob_utils.KITTI_dataloader import KITTILoader

from mob_utils.correspondece_constraint import *

import time

from config import config as cfg

from models_3d import mobilenet_v2 as nn

from tensorflow import keras
import tensorflow as tf

# DIM AVG PER CLASS (anchors)
Car = np.array([ 1.52159147,  1.64443089,  3.85813679])
Truck = np.array([  3.07392252,   2.63079903,  11.2190799 ])
Van = np.array([ 2.18928571,  1.90979592,  5.07087755])
Tram = np.array([  3.56092896,   2.39601093,  18.34125683])
Pedestrian = np.array([ 1.75554637,  0.66860882,  0.87623049])
Cyclist = np.array([ 1.73532436,  0.58028152,  1.77413709])

dims_avg = [Car , Truck, Van, Tram, Pedestrian, Cyclist]
'''
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
'''

class MobileNetV2(object):
    _default = {
        "model_path": 'model_data/3dbox_mob_v2.hdf5',
    }

    @classmethod
    def get_default(cls, n):
        if n in cls._default:
            return cls._default[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):

        self.__dict__.update(self._default) # set up default value
        self.__dict__.update(kwargs) # and update with user override

        self.generate()

    def generate(self):

        self.config = tf.ConfigProto(
        device_count={'GPU': 1},
        intra_op_parallelism_threads=1,
        allow_soft_placement=True
        )

        self.config.gpu_options.allow_growth = True
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.6

        self.sess = tf.Session(config=self.config)

        keras.backend.set_session(self.sess)

        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.hdf5'), 'Keras model or weights must be a .hdf5 file.'

        self.mobile_model = nn.network()

        try:
            self.mobile_model = load_model(model_path, compile=False)
        except:
            self.mobile_model.load_weights(self.model_path) # make sure model, anchors and classes match

        self.mobile_model._make_predict_function()


    # enter the frame, 2d_box. FOR NOW WE EXCLUDE THE CALIB FILES TO TEST ON A NORMAL VIDEO
    def predict(self, frame, image, class_nm, box_2d):

        #verify box_2d possible error
        for i in range(len(box_2d)):
            if box_2d[i] < 0:
                box_2d[i] = 0

        cl_2d = True

        # mismatch between dataset class names
        if class_nm == 'person':
            class_nm = 'pedestrian'

        if class_nm == 'bicycle':
            class_nm = 'cyclist'

        if class_nm == 'bus':
            class_nm = 'tram'

        image_data = np.array(frame, dtype='float32')

        # new 2d_box inicialization
        # left = xmin
        # top = ymin
        # right = xmax
        # bottom = ymax

        xmin, ymin, xmax, ymax = box_2d
        h = w = l = alpha = r_global = r_local = tx = ty = tz = process_time = 0

        kitti_classes = cfg().KITTI_cat


        i = 0
        # check if class belong to kitti classes
        for cl in kitti_classes:

            # string comparison (case insensitive)
            if class_nm in cl.lower():

                # not a kitti class
                cl_2d = False


                # 2D detection area
                patch = image_data[ymin : ymax, xmin : xmax]

                patch = cv2.resize(patch, (cfg().norm_h, cfg().norm_w))
                patch -= np.array([[[103.939, 116.779, 123.68]]])

                # extend it to match the training dimension
                patch = np.expand_dims(patch, 0)

                # prediction
                prediction = self.mobile_model.predict(patch)

                dim = prediction[0][0]
                bin_anchor = prediction[1][0]
                bin_confidence = prediction[2][0]

                # update with predict dimension
                dims = dims_avg[i] + dim
                h, w, l = np.array([round(dim, 2) for dim in dims])

                # update with predicted alpha, [-pi, pi]
                alpha = recover_angle(bin_anchor, bin_confidence, cfg().bin)

                # compute global and local orientation
                r_global, r_local = compute_orientation(xmax, xmin, alpha)

                # compute and update translation, (x, y, z)
                tx, ty, tz = translation_constraints(h, w, l, box_2d, r_global, r_local)

            i += 1

        return cl_2d, h, w, l, alpha, r_global, tx, ty, tz

    def close_session(self):
            self.sess.close()
