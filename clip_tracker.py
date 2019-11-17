import configparser
import os
import Queue
import threading
import time
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import chainer
import cv2
import numpy as np
import math
import copy
from PIL import Image

from predict import get_feature, get_humans_by_feature, draw_humans, create_model
from utils import parse_size

import ctypes
from ctypes import cdll

lib = cdll.LoadLibrary('./build/ssd.so')
ssd_cuda = lib.ssd

def compute_ssd(window, target, out, window_width, window_height,
                target_width, target_height, out_width, out_height, num):
    ssd_cuda(ctypes.c_void_p(window.ctypes.data),
             ctypes.c_void_p(target.ctypes.data),
             ctypes.c_void_p(out.ctypes.data),
             ctypes.c_void_p(window_width.ctypes.data),
             ctypes.c_void_p(window_height.ctypes.data),
             ctypes.c_void_p(target_width.ctypes.data),
             ctypes.c_void_p(target_height.ctypes.data),
             ctypes.c_int32(out_width), ctypes.c_int32(out_height), ctypes.c_int32(num))

QUEUE_SIZE = 5

"""
Bonus script
If you have good USB camera which gets image as well as 60 FPS,
this script will be helpful for realtime inference
"""

kx = 1.0 * 672 / 224
ky = 1.0 * 376 / 224


class Capture(threading.Thread):

    def __init__(self, cap, insize):
        super(Capture, self).__init__()
        self.cap = cap
        self.insize = insize
        self.stop_event = threading.Event()
        self.queue = Queue.Queue(QUEUE_SIZE)
        self.name = 'Capture'

    def run(self):
        while not self.stop_event.is_set():
            try:
                ret_val, image = self.cap.read()
                # only use the left half
                image = image[:, 0:672, :]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # image = cv2.resize(image, self.insize)
                self.queue.put(image, timeout=1)
            except Queue.Full:
                pass

    def get(self):
        return self.queue.get(timeout=1)

    def stop(self):
        logger.info('{} will stop'.format(self.name))
        self.stop_event.set()


class Predictor(threading.Thread):

    def __init__(self, model, cap):
        super(Predictor, self).__init__()
        self.cap = cap
        self.model = model
        self.stop_event = threading.Event()
        self.queue = Queue.Queue(QUEUE_SIZE)
        self.name = 'Predictor'
        self.is_target_set = False
        self.roi_box = [0, 0, 672, 376]
        self.target_point = [-1, -1]
        self.dist_thresh = 50

    def run(self):
        while not self.stop_event.is_set():
            try:
                image = self.cap.get()
                with chainer.using_config('autotune', True), \
                        chainer.using_config('use_ideep', 'auto'):
                    roi = cv2.resize(image[self.roi_box[1]: self.roi_box[3], self.roi_box[0]: self.roi_box[2]], self.cap.insize)
                    feature_map = get_feature(self.model, roi.transpose(2, 0, 1).astype(np.float32))
                self.queue.put((roi, feature_map), timeout=1)
            except Queue.Full:
                pass
            except Queue.Empty:
                pass

    def update_target(self, head_boxes):
        min_dist = np.inf
        nearest_box = None
        for head_box in head_boxes:
            ymin, xmin, ymax, xmax = head_box
            wroi = self.roi_box[2] - self.roi_box[0]
            hroi = self.roi_box[3] - self.roi_box[1]
            ymin = ymin * hroi / 224
            ymax = ymax * hroi / 224
            xmin = xmin * wroi / 224
            xmax = xmax * wroi / 224
            xcenter = (xmin + xmax) / 2 + self.roi_box[0]
            ycenter = (ymin + ymax) / 2 + self.roi_box[1]
            if self.get_dist([xcenter, ycenter], self.target_point) < min_dist:
                min_dist = self.get_dist([xcenter, ycenter], self.target_point)
                nearest_box = head_box

        if min_dist > self.dist_thresh:
            return

        head_box = nearest_box
        print(head_box)
        ymin, xmin, ymax, xmax = head_box
        wroi = self.roi_box[2] - self.roi_box[0]
        hroi = self.roi_box[3] - self.roi_box[1]
        ymin = ymin * hroi / 224
        ymax = ymax * hroi / 224
        xmin = xmin * wroi / 224
        xmax = xmax * wroi / 224
        xcenter = (xmin + xmax) / 2 + self.roi_box[0] # size in original image
        ycenter = (ymin + ymax) / 2 + self.roi_box[1]

        whead = xmax - xmin
        hhead = ymax - ymin

        print([ymin, xmin, ymax, xmax, wroi, hroi, whead, hhead])

        roi_size = (whead + hhead) * 1.5
        xcenter_body = xcenter
        ycenter_body = ycenter + hhead / 2
        top = ycenter_body - roi_size
        bottom = ycenter_body + roi_size
        left = xcenter_body - roi_size
        right = xcenter_body + roi_size

        print([left, top, right, bottom])

        if top < 0:
            bottom = bottom - top
            top = 0
        if bottom > 376:
            top = top - (bottom - 376)
            bottom = 376
        if left < 0:
            right = right - left
            left = 0
        if right > 672:
            left = left - (right - 672)
            right = 672
        top = int(max(top, 0))
        bottom = int(min(bottom, 376))
        left = int(max(left, 0))
        right = int(min(right, 672))

        print([left, top, right, bottom])

        # top = int(max((ymin - 1.5 * hhead) + self.roi_box[1], 0))
        # bottom = int(min((ymax + 4.5 * hhead) + self.roi_box[1], 376))
        # left = int(max((xmin - 2.5 * whead) + self.roi_box[0], 0))
        # right = int(min((xmax + 2.5 * whead) + self.roi_box[0], 672))

        curr_box = [left, top, right, bottom]
        for i in range(4):
            self.roi_box[i] = int((6 * self.roi_box[i] + curr_box[i]) / 7)
        self.target_point[0] = (6 * self.target_point[0] + xcenter) / 7
        self.target_point[1] = (6 * self.target_point[1] + ycenter) / 7
        print(self.roi_box)
        print(self.target_point)
        print('------------------------')

    def init_target(self, head_box):
        ymin, xmin, ymax, xmax = head_box
        w = xmax - xmin
        h = ymax - ymin
        xcenter = (xmin + xmax) / 2 * kx
        ycenter = (ymin + ymax) / 2 * ky
        top = int(max((ymin - 2.0 * h) * ky, 0))
        bottom = int(min((ymax + 8.0 * h) * ky, 376))
        left = int(max((xmin - 3.0 * w) * kx, 0))
        right = int(min((xmax + 3.0 * w) * kx, 672))
        self.roi_box = [left, top, right, bottom]
        self.target_point = [xcenter, ycenter]
        self.is_target_set = True
        print(self.roi_box)
        print(self.target_point)
        print('init+++++++++++++++++++++++')

    def get_dist(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) * (point1[0] - point2[0]) +
                         (point1[1] - point2[1]) * (point1[1] - point2[1]))

    def get(self):
        return self.queue.get(timeout=1)

    def stop(self):
        logger.info('{} will stop'.format(self.name))
        self.stop_event.set()


def main():
    config = configparser.ConfigParser()
    config.read('config.ini', 'UTF-8')

    model = create_model(config)

    if os.path.exists('mask.png'):
        mask = Image.open('mask.png')
        mask = mask.resize((200, 200))
    else:
        mask = None

    cap = cv2.VideoCapture('../images/left1.avi')
    if cap.isOpened() is False:
        print('Error opening video stream or file')
        exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 672)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 376)
    logger.info('camera will capture {} FPS'.format(cap.get(cv2.CAP_PROP_FPS)))

    capture = Capture(cap, model.insize)
    predictor = Predictor(model=model, cap=capture)

    capture.start()
    predictor.start()

    fps_time = 0
    degree = 0

    main_event = threading.Event()

    try:
        while not main_event.is_set() and cap.isOpened():
            degree += 5
            degree = degree % 360
            try:
                image, feature_map = predictor.get()
                humans, confidences = get_humans_by_feature(model, feature_map)
                if len(humans) and not predictor.is_target_set:
                    predictor.init_target(humans[0][0])
                elif len(humans):
                    head_boxes = []
                    for human in humans:
                        head_boxes.append(human[0])
                    predictor.update_target(head_boxes)
            except Queue.Empty:
                continue
            except Exception:
                break
            pilImg = Image.fromarray(image)
            pilImg = draw_humans(
                model.keypoint_names,
                model.edges,
                pilImg,
                humans,
                mask=mask.rotate(degree) if mask else None
            )
            img_with_humans = np.array(pilImg)
            msg = 'GPU ON' if chainer.backends.cuda.available else 'GPU OFF'
            msg += ' ' + config.get('model_param', 'model_name')
            cv2.putText(img_with_humans, 'FPS: %f' % (1.0 / (time.time() - fps_time)),
                        (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            img_with_humans = cv2.resize(img_with_humans, (224 * 3, 224 * 3))
            img_with_humans = cv2.cvtColor(img_with_humans, cv2.COLOR_RGB2BGR)
            cv2.imshow('Pose Proposal Network' + msg, img_with_humans)
            fps_time = time.time()
            # cv2.waitKey(0)
            # press Esc to exit
            if cv2.waitKey(1) == 27:
                main_event.set()
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        main_event.set()

    capture.stop()
    predictor.stop()

    capture.join()
    predictor.join()


if __name__ == '__main__':
    main()
