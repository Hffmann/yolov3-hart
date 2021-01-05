import os
import numpy as np
import cv2
import argparse

import tensorflow as tf

from tracker.hart.data import disp
from tracker.hart.data.kitti.tools import get_data
from tracker.hart.model import util
from tracker.hart.model.attention_ops import FixedStdAttention
from tracker.hart.model.eval_tools import log_norm, log_ratios, log_values, make_expr_logger
from tracker.hart.model.tracker import HierarchicalAttentiveRecurrentTracker as Hart
from tracker.hart.model.nn import AlexNetModel, IsTrainingLayer
from tracker.hart.train_tools import TrainSchedule, minimize_clipped


class HART(object):
    _default = {
        "checkpoint_path": 'tracker/checkpoints/kitti/pretrained/model.ckpt-347346',
        "norm" : 'batch',
        "alexnet_dir": 'tracker/checkpoints',
        "batch_size": 1,
        "img_size": (187, 621, 3),
        "crop_size" : (56, 56, 3),
        "rnn_units" : 100,
        "keep_prob" : .75,
        "bbox_shape" : (1, 1, 4),
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

        self.img_size, self.crop_size = [np.asarray(i) for i in (self.img_size, self.crop_size)]

        tf.reset_default_graph()
        util.set_random_seed(0)

        self.x = tf.placeholder(tf.float32, [None, self.batch_size] + list(self.img_size), name='image')
        self.y0 = tf.placeholder(tf.float32, self.bbox_shape, name='bbox')
        self.p0 = tf.ones(self.y0.get_shape()[:-1], dtype=tf.uint8, name='presence')

        self.is_training = IsTrainingLayer()
        self.builder = AlexNetModel(self.alexnet_dir, layer='conv3', n_out_feature_maps=5, upsample=False, normlayer= self.norm,
                               keep_prob=self.keep_prob, is_training=self.is_training)

        self.model = Hart(self.x, self.y0, self.p0, self.batch_size, self.crop_size, self.builder, self.rnn_units,
                     bbox_gain=[-4.78, -1.8, -3., -1.8],
                     zoneout_prob=(.05, .05),
                     normalize_glimpse=True,
                     attention_module=FixedStdAttention,
                     debug=True,
                     transform_init_features=True,
                     transform_init_state=True,
                     dfn_readout=True,
                     feature_shape=(14, 14),
                     is_training=self.is_training)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, self.checkpoint_path)
        self.model.test_mode(self.sess)

        self.tensors = [self.model.pred_bbox, self.model.att_pred_bbox, self.model.glimpse, self.model.obj_mask]



    def bbox_to_hart(self, bboxes, shape, mode):

        if mode == "det":
            bboxes_result = self.track_bboxes =  bboxes
        else:
            bboxes_result = self.track_bboxes

        for i in range(len(bboxes)):

            x_hart = self.img_size[0]
            y_hart = self.img_size[1]

            x_ = shape[0]
            y_ = shape[1]

            x_scale = x_hart/x_
            y_scale = y_hart/y_

            if mode == "det":

                bboxes_result[i][0] = (bboxes[i][0] * y_scale)
                bboxes_result[i][1] = (bboxes[i][1] * x_scale)
                bboxes_result[i][2] = (bboxes[i][2] * y_scale) - bboxes[i][0]
                bboxes_result[i][3] = (bboxes[i][3] * x_scale) - bboxes[i][1]

            else:

                bboxes_result[i][0] = bboxes[i][1, 0, 0, 0] * y_scale
                bboxes_result[i][1] = bboxes[i][1, 0, 0, 1] * x_scale
                bboxes_result[i][2] = (bboxes[i][1, 0, 0, 2] - bboxes[i][1, 0, 0, 0]) * y_scale
                bboxes_result[i][3] = (bboxes[i][1, 0, 0, 3] - bboxes[i][1, 0, 0, 1]) * x_scale

        return bboxes_result


    def bbox_to_mob(self, bboxes, shape):

        for i in range(len(bboxes)):

            x_hart = self.img_size[0]
            y_hart = self.img_size[1]

            x_ = shape[0]
            y_ = shape[1]

            x_scale = x_hart/x_
            y_scale = y_hart/y_

            bboxes[i][1, 0, 0, 0] =  bboxes[i][1, 0, 0, 0] / y_scale
            bboxes[i][1, 0, 0, 1] =  bboxes[i][1, 0, 0, 1] / x_scale
            bboxes[i][1, 0, 0, 2] =  (bboxes[i][1, 0, 0, 2] / y_scale) + bboxes[i][1, 0, 0, 0]
            bboxes[i][1, 0, 0, 3] =  (bboxes[i][1, 0, 0, 3] / y_scale) + bboxes[i][1, 0, 0, 1]

        return bboxes

    def pred_track(self, imgs, bboxes):
        
        pred_bboxes = pred_bbox = pred_att = glimpse = obj_mask = []

        for i in range(len(bboxes)):
            pred_bbox, pred_att, glimpse, obj_mask = self.sess.run(
                self.tensors, feed_dict = {self.x: imgs, self.y0: np.reshape(bboxes[i], self.bbox_shape)})

            pred_bboxes.append(pred_bbox)

        return pred_bboxes

    def close_session(self):
            self.sess.close()
