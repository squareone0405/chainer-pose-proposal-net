#!/usr/bin/python2

import rospy
import std_msgs
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from ppn.msg import Skeleton
from cv_bridge import CvBridge

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
from PIL import Image

from predict import get_feature, get_humans_by_feature, draw_humans, create_model
from utils import parse_size

import pyzed.sl as sl

QUEUE_SIZE = 5

"""
Bonus script
If you have good USB camera which gets image as well as 60 FPS,
this script will be helpful for realtime inference
"""


class Capture(threading.Thread):
    def _init_zed(self):
        self.zed = sl.Camera()
        self.init = sl.InitParameters()
        self.init.camera_resolution = sl.RESOLUTION.RESOLUTION_VGA
        self.init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_PERFORMANCE
        self.init.coordinate_units = sl.UNIT.UNIT_METER

        err = self.zed.open(self.init)
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(err))
            self.zed.close()
            exit(1)

        self.runtime = sl.RuntimeParameters()
        self.runtime.sensing_mode = sl.SENSING_MODE.SENSING_MODE_STANDARD

    def __init__(self, insize):
        super(Capture, self).__init__()
        self._init_zed()
        self.depth = None
        self.insize = insize
        self.kx = 1.0 * self.zed.get_resolution().width / insize[0]
        self.ky = 1.0 * self.zed.get_resolution().height / insize[1]
        self.stop_event = threading.Event()
        self.queue = Queue.Queue(QUEUE_SIZE)
        self.name = 'Capture'

    def run(self):
        image_size = self.zed.get_resolution()
        width_zed = image_size.width
        height_zed = image_size.height
        print("ZED will capture at %d x %d" % (width_zed, height_zed))
        image_zed = sl.Mat(width_zed, height_zed, sl.MAT_TYPE.MAT_TYPE_8U_C4)
        depth_zed = sl.Mat()
        while not self.stop_event.is_set():
            try:
                err = self.zed.grab(self.runtime)
                if err == sl.ERROR_CODE.SUCCESS:
                    self.zed.retrieve_image(image_zed, sl.VIEW.VIEW_LEFT, sl.MEM.MEM_CPU, int(width_zed),
                                            int(height_zed))
                    image = image_zed.get_data()
                    self.zed.retrieve_measure(depth_zed, sl.MEASURE.MEASURE_DEPTH)
                    self.depth = depth_zed.get_data()
                    cv2.imshow('CV', self.depth)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, self.insize)
                    self.queue.put(image, timeout=1)
            except Queue.Full:
                pass

    def get(self):
        return self.queue.get(timeout=1)

    def stop(self):
        logger.info('{} will stop'.format(self.name))
        self.zed.close()
        self.stop_event.set()


class Predictor(threading.Thread):

    def __init__(self, model, cap):
        super(Predictor, self).__init__()
        self.cap = cap
        self.model = model
        self.stop_event = threading.Event()
        self.queue = Queue.Queue(QUEUE_SIZE)
        self.name = 'Predictor'

    def run(self):
        while not self.stop_event.is_set():
            try:
                image = self.cap.get()
                with chainer.using_config('autotune', True), \
                        chainer.using_config('use_ideep', 'auto'):
                    feature_map = get_feature(self.model, image.transpose(2, 0, 1).astype(np.float32))
                self.queue.put((image, feature_map), timeout=1)
            except Queue.Full:
                pass
            except Queue.Empty:
                pass

    def get(self):
        return self.queue.get(timeout=1)

    def stop(self):
        logger.info('{} will stop'.format(self.name))
        self.stop_event.set()

    def pub_skeleton(self, left_humans):
        msg = Skeleton()
        msg.header = std_msgs.msg.Header()
        msg.header.stamp = rospy.Time.now()
        msg.human_num = len(left_humans)
        if len(left_humans) > 0:
            skeleton_num = [0] * len(left_humans)
            types = []
            points = []
            for i in range(len(left_humans)):
                skeleton_num[i] = len(left_humans[i]) - 1
                for key in sorted(left_humans[i]):
                    if key != 0:
                        ymin, xmin, ymax, xmax = left_humans[i][key]
                        p = Point()
                        p.x = 0.5 * (xmin + xmax) * self.cap.kx
                        p.y = 0.5 * (ymin + ymax) * self.cap.ky
                        p.z = 0.0
                        temp_z = np.nan_to_num(self.cap.depth[int(p.y), int(p.x)])
                        print(temp_z)
                        if temp_z > 0:
                            p.z = temp_z
                        types.append(key)
                        points.append(p)
            msg.skeleton_num = skeleton_num
            msg.types = types
            msg.points = points
            pub.publish(msg)

def main():
    config = configparser.ConfigParser()
    config.read('config.ini', 'UTF-8')

    model = create_model(config)

    if os.path.exists('mask.png'):
        mask = Image.open('mask.png')
        mask = mask.resize((200, 200))
    else:
        mask = None

    cap = cv2.VideoCapture(0)
    if cap.isOpened() is False:
        print('Error opening video stream or file')
        exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    logger.info('camera will capture {} FPS'.format(cap.get(cv2.CAP_PROP_FPS)))

    capture = Capture(model.insize)
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
                humans = get_humans_by_feature(model, feature_map)
            except Queue.Empty:
                continue
            except Exception:
                break
            predictor.pub_skeleton(humans)
            pilImg = Image.fromarray(image)
            pilImg = draw_humans(
                model.keypoint_names,
                model.edges,
                pilImg,
                humans,
                mask=mask.rotate(degree) if mask else None
            )
            img_with_humans = cv2.cvtColor(np.asarray(pilImg), cv2.COLOR_RGB2BGR)
            msg = 'GPU ON' if chainer.backends.cuda.available else 'GPU OFF'
            msg += ' ' + config.get('model_param', 'model_name')
            cv2.putText(img_with_humans, 'FPS: %f' % (1.0 / (time.time() - fps_time)),
                        (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            img_with_humans = cv2.resize(img_with_humans, (3 * model.insize[0], 3 * model.insize[1]))
            cv2.imshow('Pose Proposal Network' + msg, img_with_humans)
            fps_time = time.time()
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
    rospy.init_node('skeleton_pub', anonymous=True)
    pub = rospy.Publisher('skeleton', Skeleton, queue_size=1000)
    main()
