#!/usr/bin/python2

import rospy
import std_msgs
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from ppn.msg import Skeleton
from cv_bridge import CvBridge

import configparser
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import os
import time

import chainer
from chainercv.utils import non_maximum_suppression
import cv2
import numpy as np

from predict import COLOR_MAP
from predict import estimate, draw_humans, create_model
from utils import parse_size


config = configparser.ConfigParser()
config.read('config.ini', 'UTF-8')
model = create_model(config)
bridge = CvBridge()

def image_callback(image_msg):
    image = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
    # image = cv2.imread('city-walk.png')
    shape_ori = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, model.insize)
    with chainer.using_config('autotune', True):
        humans = estimate(model,
                          image.transpose(2, 0, 1).astype(np.float32))
    print(humans)
    msg = Skeleton()
    msg.header = std_msgs.msg.Header()
    msg.header.stamp = rospy.Time.now()
    msg.human_num = len(humans)
    if len(humans) > 0:
        kx = shape_ori[1] * 1.0 / model.insize[1]
        ky = shape_ori[0] * 1.0 / model.insize[0]
        skeleton_num = [0]*len(humans)
        types = []
        points = []
        for i in range(len(humans)):
            skeleton_num[i] = len(humans[i]) - 1
            for key in sorted(humans[i]):
                if key != 0:
                    ymin, xmin, ymax, xmax = humans[i][key]
                    p = Point()
                    p.x = (xmin + xmax) * kx / 2
                    p.y = (ymin + ymax) * ky / 2
                    p.z = 0
                    types.append(key)
                    points.append(p)
        msg.skeleton_num = skeleton_num
        msg.types = types
        msg.points = points
    pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('skeleton_pub', anonymous=True)
    pub = rospy.Publisher('skeleton', Skeleton, queue_size=1000)
    sub = rospy.Subscriber('mv_26801267/image_raw', Image, image_callback)
    rospy.spin()

