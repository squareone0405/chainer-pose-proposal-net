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

cx = 336.162
cy = 184.095
fx = 350.003
fy = 350.003
k1 = -0.17211
k2 = 0.0267822
K = np.zeros((3, 3))
K[0, 0] = fx
K[1, 1] = fy
K[0, 2] = cx
K[1, 2] = cy
K[2, 2] = 1.0

BaseLine = 119.971
CV_VGA = 0.0129813
RX_VGA = -0.000799271
RZ_VGA = -5.96179e-5
T = np.array([-BaseLine, 0, 0])
Rz, _ = cv2.Rodrigues(np.array([0, 0, RZ_VGA]))
Ry, _ = cv2.Rodrigues(np.array([0, CV_VGA, 0]))
Rx, _ = cv2.Rodrigues(np.array([RX_VGA, 0, 0]))
R = np.dot(Rz, np.dot(Ry, Rx))

def compute_depth():
    pass

def get_points(human, kx, ky):
    types = []
    points = []
    for key in sorted(human):
        if key != 0:
            ymin, xmin, ymax, xmax = human[key]
            p = Point()
            p.x = (xmin + xmax) * kx / 2
            p.y = (ymin + ymax) * ky / 2
            p.z = 0
            types.append(key)
            points.append(p)
    return points, types

def image_callback(image_msg):
    # image = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
    # image = cv2.imread('city-walk.png')

    '''shape_ori = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, model.insize)

    left_image = cv2.imread('1_l.png')
    right_image = cv2.imread('1_r.png')

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
        skeleton_num = [0] * len(humans)
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
    pub.publish(msg)'''

    left_image = cv2.imread('lwb_l.png')
    right_image = cv2.imread('lwb_r.png')

    stereo = cv2.StereoBM_create(16, 25)
    disparity = stereo.compute(cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY),
                               cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY))
    depth = (700.0 * 120.0) / (disparity + 0.1)

    shape_ori = left_image.shape

    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
    left_image = cv2.resize(left_image, model.insize)
    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
    right_image = cv2.resize(right_image, model.insize)

    with chainer.using_config('autotune', True):
        print()
        left_humans = estimate(model, left_image.transpose(2, 0, 1).astype(np.float32))
        right_humans = estimate(model, right_image.transpose(2, 0, 1).astype(np.float32))
    print('-' * 10 + 'left' + '-' * 10)
    print(left_humans)
    print('+' * 10 + 'right' + '+' * 10)
    print(right_humans)

    kx = shape_ori[1] * 1.0 / model.insize[1]
    ky = shape_ori[0] * 1.0 / model.insize[0]

    msg = Skeleton()
    msg.header = std_msgs.msg.Header()
    msg.header.stamp = rospy.Time.now()
    msg.human_num = len(left_humans)
    if len(left_humans) > 0:
        skeleton_num = [0] * len(left_humans)
        left_types = []
        left_points = []
        right_types = []
        right_points = []
        for i in range(len(left_humans)):
            skeleton_num[i] = len(left_humans[i]) - 1
            left_points, left_types = get_points(left_humans[i], kx, ky)
        for i in range(len(right_humans)):
            right_points, right_types = get_points(right_humans[i], kx, ky)
        for i in range(len(left_points)):
            pass
        msg.skeleton_num = skeleton_num
        msg.types = left_types
        msg.points = left_points
    pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('skeleton_pub', anonymous=True)
    pub = rospy.Publisher('skeleton', Skeleton, queue_size=1000)
    sub = rospy.Subscriber('mv_26801267/image_raw', Image, image_callback)
    rospy.spin()

