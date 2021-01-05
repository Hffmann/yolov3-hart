########################################################################################
#
# Hierarchical Attentive Recurrent Tracking
# Copyright (C) 2017  Adam R. Kosiorek, Oxford Robotics Institute, University of Oxford
# email:   adamk@robots.ox.ac.uk
# webpage: http://ori.ox.ac.uk
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
########################################################################################
import tensorflow as tf

from tracker.neurocity.component.model import base


class Model(base.BaseModel):
    """An abstraction of a Neural Network model. It simplifies layer management and supports
     train/test modes. To define a new model, derive a class and overload the _build method,
     which should be responsible for constructing the model, e.g.:

        class MLP(object):
            def _build(self):
                self.inpt = tf.placeholder(tf.float32, (32, 100), name='inpt')
                l1 = AffineLayer(self.inpt, 200)
                l2 = AffineLayer(l1, 10)

    Both layers will be registered in model.layers attribute.

    It is important to call:
        model.train_mode() - before training
        model.test_mode() - before testing

    Note: `mode` can be overwritten by calling one of the global setters:
        neurocity.train_mode() or neurocity.test_mode()"""

    def __init__(self, name='Model'):
        super(Model, self).__init__()

        self.name = name
        self.layers = []

        model_vars = set(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))
        trainable_vars = set(tf.trainable_variables())
        with tf.variable_scope(self.name):
            with self:
                self._build()
            base.get_model().register(self)

        self.trainable_vars = set(tf.trainable_variables()) - trainable_vars
        self.model_vars = set(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)) - model_vars - self.trainable_vars

    @property
    def vars(self):
        return self.trainable_vars.union(self.model_vars)

    def saver(self, **kwargs):
        """Returns a Saver for all (trainable and model) variables used by the model.
        Model variables include e.g. moving mean and average in BatchNorm.

        :return: tf.Saver
        """

        return tf.train.Saver(self.vars, **kwargs)

    def _build(self):
        raise NotImplementedError

    def __enter__(self):
        base.set_model(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        base.reset_model()
        if exc_type is None:
            return True
