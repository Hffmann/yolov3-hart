# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import tensorflow as tf

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image
import cv2

from yolo3.model import yolo_eval, yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

from visualization3Dbox import draw_3Dbox


YOLO_time = []
HART_time = []
YOLO_det = []
HART_det = []
ALL_time = []
ALL_det = []

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "max_output_size" : 25,
        "score" : 0.5,
        "iou" : 0.5,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names, self.class_enumeration = self._get_class()

        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        class_enumeration = list(enumerate(class_names, 0))

        return class_names, class_enumeration

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]

        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype="uint8")

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)

        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)

        return boxes, scores, classes



    def detect_image(self, mobilenet_v2, frame, image, mode):

# YOLO TIME
############################################################################################
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension

        # FEED IMAGE TO YOLO.
        with self.sess.as_default():
            with self.sess.graph.as_default():
                out_boxes, out_scores, out_classes = self.sess.run(
                    [self.boxes, self.scores, self.classes],
                    feed_dict={
                        self.yolo_model.input: image_data,
                        self.input_image_shape: [image.size[1], image.size[0]],
                        K.learning_phase(): 0
                    })

        # FOR EACH (initialize box parameters)
        boxes_nms = []
        scores_nms = []
        classes_nms = []
        classes_colors = []

        for i, c in reversed(list(enumerate(out_classes))):

            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            boxes_nms.append([int(top), int(left), int(bottom), int(right)])
            scores_nms.append(float(score))
            classes_nms.append(predicted_class)

        # aplicação de supressão não-máxima
        nms_index = cv2.dnn.NMSBoxes(boxes_nms, scores_nms, self.score, self.iou)
        # aplicação de supressão não-máxima

        end = timer()

        print("YOLO Processing time: {}".format(end - start))

        YOLO_time.append(end - start)
        ALL_time.append(end - start)
        YOLO_det.append(len(nms_index))
        ALL_det.append(len(nms_index))
############################################################################################

        print('{} corrected boxes for {}'.format(len(nms_index), 'img'))

        # verificar se existe ao menos uma detecção
        if len(nms_index) > 0:

        	# laço para desenho de cada caixa de objetos
        	for i in nms_index.flatten():

        		                # extração das coordenadas das caixas
                                (t, l) = (boxes_nms[i][0], boxes_nms[i][1])
                                (b, r) = (boxes_nms[i][2], boxes_nms[i][3])

                                # verificar para cada classe e valor
                                for color_idx, val in self.class_enumeration:
                                    # comparar se a classe é a mesma e definir o valor cor da classe
                                                    if val == classes_nms[i]:
                                                                class_color = color_idx
                                                                break    # break here

                                # (DRAWING)
                                color = [int(c) for c in self.colors[class_color]]
                                label = "{}: {:.2f}".format(classes_nms[i], scores_nms[i])

                                if mode == "3d":
                                    #3d prediction from 2d box
                                    box_2d = np.array([l, t, r, b])
                                    with mobilenet_v2.sess.as_default():
                                        with mobilenet_v2.sess.graph.as_default():
                                            class_2d, height, width, length, alpha, rot_global, tx, ty, tz = mobilenet_v2.predict(
                                                frame, image, classes_nms[i], box_2d)

                                    # 3D drawing for kitti classes
                                    draw_3Dbox(frame, color, box_2d, rot_global, height, width, length, tx, ty, tz)

                                    # 2D drawing for the other classes
                                    if class_2d:
                                        cv2.rectangle(frame, (l, t), (r, b), color, 2)

                                    # Print class and box data
                                    print(label, (l, t), (r, b), (height, width, length, alpha))

                                else:
                                    cv2.rectangle(frame, (l, t), (r, b), color, 2)
                                    # Print class and box data
                                    print(label, (l, t), (r, b))

                                cv2.putText(frame, label, (l, t - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                classes_colors.append(color)

        return frame, boxes_nms, nms_index, classes_nms, scores_nms

    def close_session(self):
            self.sess.close()


def show_image(frame, mode, fps, accum_time, curr_fps, prev_time, isOutput, out):

    txt_mode = "Mode: " + mode
    result = np.asarray(frame)
    curr_time = timer()
    exec_time = curr_time - prev_time
    prev_time = curr_time
    accum_time = accum_time + exec_time
    curr_fps = curr_fps + 1
    if accum_time > 1:
        accum_time = accum_time - 1
        fps = "FPS: " + str(curr_fps)
        curr_fps = 0

    cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.50, color=(255, 0, 0), thickness=2)

    cv2.putText(result, text=txt_mode, org=(3, 370), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(255, 0, 0), thickness=2)

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result", result)
    if isOutput:
        out.write(result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

    return accum_time, curr_fps, prev_time, fps, out


def prep_run(yolo, hart, video_path):

    print("Running preparation")
    start = timer()

    prep_vid = cv2.VideoCapture(video_path)
    if not prep_vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(prep_vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = prep_vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(prep_vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(prep_vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    return_value, frame = prep_vid.read()

    image = Image.fromarray(frame)

#PREP DET
############################################################################################################
    if yolo.model_image_size != (None, None):
        assert yolo.model_image_size[0]%32 == 0, 'Multiples of 32 required'
        assert yolo.model_image_size[1]%32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(yolo.model_image_size)))
    else:
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension

    # FEED IMAGE TO YOLO.
    with yolo.sess.as_default():
        with yolo.sess.graph.as_default():
            out_boxes, out_scores, out_classes = yolo.sess.run(
                [yolo.boxes, yolo.scores, yolo.classes],
                feed_dict={
                    yolo.yolo_model.input: image_data,
                    yolo.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })
############################################################################################################
#PREP TRACK
############################################################################################################
    prev_frame = frame

    return_value, frame = prep_vid.read()

    hart_frame = cv2.resize(frame, (hart.img_size[1], hart.img_size[0]))
    prev_frame = cv2.resize(prev_frame, (hart.img_size[1], hart.img_size[0]))

    imgs = np.empty([2, 1] + list(hart.img_size), dtype=np.float32)

    imgs[0, 0] = Image.fromarray(prev_frame)
    imgs[1, 0] = Image.fromarray(hart_frame)

    hart_bboxes = hart.bbox_to_hart(out_boxes, frame.shape, mode = "det")

    track_bboxes = hart.pred_track(imgs, hart_bboxes)
############################################################################################################
    end = timer()
    print("Preparation completed in {} s".format(end-start))



def time_eval(yolo, mobilenet_v2, hart, track, switch):

    time_accum = []
    if len(HART_time) == 0:
        print("Tempo medio YOLO: {} s".format(sum(YOLO_time)/len(YOLO_time)))
    else:
        print("Tempo medio YOLO: {} s | Tempo medio HART: {} s".format(sum(YOLO_time)/len(YOLO_time), sum(HART_time)/len(HART_time)))
    print("Tempo total: {} s | Tempo total YOLO: {} s | Tempo total HART: {} s".format((sum(YOLO_time) + sum(HART_time)), sum(YOLO_time), sum(HART_time)))
    print("Numero medio de detecções por frame: {} | Numero de frames processados: {}".format((sum(YOLO_det) + sum(HART_det))/(len(YOLO_det) + len(HART_det)), len(YOLO_det) + len(HART_det)))
    print("Numero de frames YOLO: {} | Numero de frames HART: {}".format(len(YOLO_time), len(HART_time)))
    for i in range(len(ALL_time)):

        if i == 0:
            time_accum.append(ALL_time[i])
        else:
            time_accum.append(ALL_time[i] + time_accum[i-1])

    print("Frames: {}".format(len(YOLO_time)))
    print(len(time_accum))

    c1 = c2 = 0
    for i in range((len(ALL_det))):
        if ALL_det[i] > 6:
            c1 += 1
        else:
            c2 += 1
    print("Higher than 6: {} | Less than or equal to 6: {}".format(c1, c2) )

    with open("time_eval/track_{}_switch_{}_time.txt".format(track, switch), "w") as text_file:

        for i in range(len(time_accum)):
            print("{}".format(time_accum[i]), file=text_file)

    with open("time_eval/track_{}_switch_{}_det.txt".format(track, switch), "w") as text_file:

        for i in range(len(ALL_det)):
            print("{}".format(ALL_det[i]), file=text_file)

    with open("time_eval/track_{}_switch_{}_info.txt".format(track, switch), "w") as text_file:

        if len(HART_time) == 0:
            print("Tempo medio YOLO: {} s".format(sum(YOLO_time)/len(YOLO_time)), file=text_file)
        else:
            print("Tempo medio YOLO: {} s | Tempo medio HART: {} s".format(sum(YOLO_time)/len(YOLO_time), sum(HART_time)/len(HART_time)), file=text_file)
        print("Tempo total: {} s | Tempo total YOLO: {} s | Tempo total HART: {} s".format((sum(YOLO_time) + sum(HART_time)), sum(YOLO_time), sum(HART_time)), file=text_file)
        print("Numero medio de detecções por frame: {} | Numero de frames processados: {}".format((sum(YOLO_det) + sum(HART_det))/(len(YOLO_det) + len(HART_det)), len(YOLO_det) + len(HART_det)), file=text_file)
        print("Numero de frames YOLO: {} | Numero de frames HART: {}".format(len(YOLO_time), len(HART_time)), file=text_file)

    yolo.close_session()
    mobilenet_v2.close_session()
    hart.close_session()


def detect_video(yolo, mobilenet_v2, hart, mode, switch, track_idx, video_path, output_path = ""):

    import cv2

    prep_run(yolo, hart, video_path)

    inicio = timer()

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    isOutput = True if output_path != "" else False

    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    else:
        out = 0

    fps = "FPS: ??"
    accum_time = 0
    curr_fps = 0
    prev_time = timer()
    system = "Detect"

    while True:

        if system == "Track":
            for i in range(track_idx):
                prev_frame = frame
                return_value, frame = vid.read()

                print(system, return_value)
                if not return_value:
                    final = timer()
                    print("Tempo: {} s".format(final - inicio))
                    time_eval(yolo, mobilenet_v2, hart, track_idx, switch)

                draw_frame = frame

# HART TIME
############################################################################################
                start = timer()

                hart_frame = cv2.resize(frame, (hart.img_size[1], hart.img_size[0]))
                prev_frame = cv2.resize(prev_frame,  (hart.img_size[1], hart.img_size[0]))

                imgs = np.empty([2, 1] + list(hart.img_size), dtype=np.float32)

                imgs[0, 0] = Image.fromarray(prev_frame)
                imgs[1, 0] = Image.fromarray(hart_frame)

                if i == 0:
                    hart_bboxes = hart.bbox_to_hart(bboxes, frame.shape, mode = "det")
                else:
                    hart_bboxes = hart.bbox_to_hart(mob_bboxes, frame.shape, mode = "track")

                track_bboxes = hart.pred_track(imgs, hart_bboxes)

                #DRAWING PREPARATION
                mob_bboxes = hart.bbox_to_mob(track_bboxes, frame.shape)

                end = timer()

                print("HART Processing time: {}".format(end - start))
                HART_time.append(end - start)
                ALL_time.append(end - start)
                HART_det.append(len(idx))
                ALL_det.append(len(idx))
############################################################################################

                #DRAWING
                # verificar se existe ao menos uma detecção
                if len(idx) > 0:
                    # extração das coordenadas das caixas
                    for i in idx.flatten():

                        # laço para desenho de cada caixa de objetos
                        (t, l) = (int(mob_bboxes[i][1, 0, 0, 0]), int(mob_bboxes[i][1, 0, 0, 1]))
                        (b, r) = (int(mob_bboxes[i][1, 0, 0, 2]), int(mob_bboxes[i][1, 0, 0, 3]))

                        # verificar para cada classe e valor
                        for color_idx, val in yolo.class_enumeration:
                            # comparar se a classe é a mesma e definir o valor cor da classe
                                            if val == classes[i]:
                                                        class_color = color_idx
                                                        break    # break here

                        # (DRAWING)
                        color = [int(c) for c in yolo.colors[class_color]]

                        if mode == "3d":
                            #3d prediction from 2d box
                            box_2d = np.array([l, t, r, b])
                            with mobilenet_v2.sess.as_default():
                                with mobilenet_v2.sess.graph.as_default():
                                    class_2d, height, width, length, alpha, rot_global, tx, ty, tz = mobilenet_v2.predict(
                                        draw_frame, imgs[1, 0], classes[i], box_2d)

                            # 3D drawing for kitti classes
                            draw_3Dbox(draw_frame, color, box_2d, rot_global, height, width, length, tx, ty, tz)

                            # 2D drawing for the other classes
                            if class_2d:
                                cv2.rectangle(draw_frame, (l, t), (r, b), color, 2)

                        else:
                            cv2.rectangle(draw_frame, (l, t), (r, b), color, 2)

                        label = "{}: {:.2f}".format(classes[i], scores[i])
                        cv2.putText(draw_frame, label, (l, t - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                accum_time, curr_fps, prev_time, fps, out = show_image(
                    draw_frame, system, fps, accum_time, curr_fps, prev_time, isOutput, out)

            system = "Detect"

        if system == "Detect":

            return_value, frame = vid.read()

            print(system, return_value)
            if not return_value:
                final = timer()
                print("Tempo: {} s".format(final - inicio))
                time_eval(yolo, mobilenet_v2, hart, track_idx, switch)

            image = Image.fromarray(frame)

            det_frame, bboxes, idx, classes, scores = yolo.detect_image(mobilenet_v2, frame, image, mode)
            accum_time, curr_fps, prev_time, fps, out = show_image(
                det_frame, system, fps, accum_time, curr_fps, prev_time, isOutput, out)

            try:
                switch = int(switch)
                if len(idx) <= switch:
                    system = "Track"
                else:
                    system = "Detect"
            except:
                system = "Track"

    yolo.close_session()
