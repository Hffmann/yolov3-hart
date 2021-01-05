
import sys
import argparse
#modify to yolo
from yolo import YOLO, detect_video

from mobilenet import MobileNetV2

from hart import HART

#modify to yolo
from PIL import Image

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    ap = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    ap.add_argument('--model', type=str,
            help='path to model weight file, default ' + YOLO.get_defaults("model_path"))
    ap.add_argument('--stereo', type=str,
            help='path to model weight file, default ' + YOLO.get_defaults("model_path"))
    ap.add_argument('--mode', type=str, default = "2d",
            help='2d or 3d')
    ap.add_argument('--track', type=int, default = "1",
            help='number of final detections to switch between detection and tracking strategies')
    ap.add_argument('--switch', type=str, default = "6",
            help='number of final detections to switch between detection and tracking strategies')
    ap.add_argument('--anchors', type=str,
            help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path"))
    ap.add_argument('--classes', type=str,
            help='path to class definitions, default ' + YOLO.get_defaults("classes_path"))
    ap.add_argument('--gpu_num', type=int,
            help='number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num")))
    ap.add_argument("--input", nargs='?', type=str, required=False, default='./video_input/0010.mp4',
            help = "video input path")
    ap.add_argument("--output", nargs='?', type=str, default="",
            help = "[Optional] video output path")

    FLAGS = ap.parse_args()

    detect_video(YOLO(**vars(FLAGS)), MobileNetV2(**vars(FLAGS)), HART(**vars(FLAGS)), FLAGS.mode, FLAGS.switch, FLAGS.track, FLAGS.input, FLAGS.output)
