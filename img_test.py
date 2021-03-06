import matplotlib.pyplot as plt

from scipy import signal
import math

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
from PIL import ImageDraw, Image

from predict import COLOR_MAP
from predict import estimate, draw_humans, create_model
from utils import parse_size

# VGA on zed 1
cx = 336.162
cy = 184.095
fx = 350.003
fy = 350.003
# HD on zed 1
'''cx = 642.325
cy = 354.191
fx = 700.006
fy = 700.006'''
# FHD on zed1
'''cx = 967.65
cy = 531.382
fx = 1400.01
fy = 1400.01'''

BaseLine = 0.12
plot = True


def pix2cam(point):
    x = point[2] * (point[0] - cx) / fx
    y = point[2] * (point[1] - cy) / fy
    z = point[2]
    return [x, y, z]


def get_points(human, image_left, image_right, kx, ky):
    FocalLength = fx

    points = np.zeros((len(human) - 1, 3), dtype='float64')
    mask_size = 3
    interp_range = (0.1, 1)

    start_time = time.time()
    image_left_ori = cv2.cvtColor(image_left, cv2.COLOR_RGB2GRAY)
    image_right_ori = cv2.cvtColor(image_right, cv2.COLOR_RGB2GRAY)
    kx_ori = kx
    ky_ori = ky

    scale_ratio = 8 if image_left.shape[1] > 1000 else 4
    image_left = cv2.resize(image_left_ori, (int(image_left_ori.shape[1] / scale_ratio),
                                         int(image_left_ori.shape[0] / scale_ratio)), cv2.INTER_CUBIC)
    image_right = cv2.resize(image_right_ori, (int(image_right_ori.shape[1] / scale_ratio),
                                         int(image_right_ori.shape[0] / scale_ratio)), cv2.INTER_CUBIC)

    if plot is True:
        plt.imshow(image_left, cmap='gray')
        plt.show()
        plt.imshow(image_right, cmap='gray')
        plt.show()

    kx = kx_ori / scale_ratio
    ky = ky_ori / scale_ratio

    search_xmin = -20
    search_xmax = 0
    search_ymin = -2
    search_ymax = 2

    window_all = np.array([], dtype='int32')
    target_all = np.array([], dtype='int32')
    window_width = np.zeros((len(human) - 1), dtype='int32')
    window_height = np.zeros((len(human) - 1), dtype='int32')
    target_width = np.zeros((len(human) - 1), dtype='int32')
    target_height = np.zeros((len(human) - 1), dtype='int32')

    ssd = np.zeros((len(human) - 1, search_ymax - search_ymin + 1, search_xmax - search_xmin + 1), dtype='int32')
    idx = 0
    for key in sorted(human):
        if key != 0:
            ymin_f, xmin_f, ymax_f, xmax_f = human[key] # TODO change the range of head and hip

            if key == 1: # head top
                ymin_f = ymin_f * 0.7 + ymax_f * 0.3
            if key in [9, 10]: # left hip or right hip
                xmin_f = xmin_f - (xmax_f - xmin_f) * 0.3
                xmax_f = xmax_f + (xmax_f - xmin_f) * 0.3

            ymin = max(int(ymin_f * ky), 0)
            xmin = max(int(xmin_f * kx), 0)
            ymax = min(int(ymax_f * ky), image_left.shape[0])
            xmax = min(int(xmax_f * kx), image_left.shape[1])

            window = image_left[ymin:ymax, xmin:xmax].astype(np.int32)
            # print(ymin, ymax, xmin, xmax)

            target = np.full((window.shape[0] + search_ymax - search_ymin,
                              window.shape[1] + search_xmax - search_xmin), 512, dtype='int32')
            target_ymin = max(ymin + search_ymin, 0)
            target_ymax = min(ymax + search_ymax, image_right.shape[0])
            target_xmin = max(xmin + search_xmin, 0)
            target_xmax = min(xmax + search_xmax, image_right.shape[1])
            target_from_right = image_right[target_ymin:target_ymax, target_xmin:target_xmax]
            clip_ymin = max(-(ymin + search_ymin), 0)
            clip_ymax = max(ymax + search_ymax - image_right.shape[0], 0)

            target[clip_ymin: target.shape[0] - clip_ymax, target.shape[1] - (target_xmax - target_xmin):] = \
                target_from_right

            window_all = np.append(window_all, window.flatten())
            target_all = np.append(target_all, target.flatten())
            window_width[idx] = window.shape[1]
            window_height[idx] = window.shape[0]
            target_width[idx] = target.shape[1]
            target_height[idx] = target.shape[0]
            idx = idx + 1

    print('cpu:' + str(time.time() - start_time))
    out = np.zeros_like(ssd, dtype='int32')
    compute_ssd(window_all, target_all, out, window_width, window_height,
                target_width, target_height, ssd.shape[2], ssd.shape[1], len(human) - 1)
    ssd = out
    print('cuda ssd:' + str(time.time() - start_time))

    heat_map = ssd.sum(axis=0)

    offset_window = 0
    offset_target = 0
    if plot is True:
        for i in range(len(human) - 1):
            plt.subplot(311)
            plt.imshow(window_all[offset_window: offset_window + window_width[i] * window_height[i]]
                       .reshape((window_height[i], window_width[i])), cmap='gray')
            plt.subplot(312)
            plt.imshow(target_all[offset_target: offset_target + target_width[i] * target_height[i]]
                       .reshape((target_height[i], target_width[i])), cmap='gray')
            plt.subplot(313)
            plt.imshow(ssd[i], cmap='gray')
            plt.show()
            offset_window += window_width[i] * window_height[i]
            offset_target += target_width[i] * target_height[i]
        plt.imshow(heat_map, cmap='gray')
        plt.show()

    mask = np.ones((mask_size, mask_size), dtype='int32')
    convolved = signal.convolve2d(heat_map, mask, 'valid')
    min_idx = np.argmin(convolved.reshape(-1))
    center_x, center_y = min_idx % convolved.shape[1] + 1, min_idx / convolved.shape[1] + 1
    # print(center_x, center_y)

    offset_x = (search_xmax - search_xmin - center_x) * scale_ratio
    offset_y = (search_ymin + center_y) * scale_ratio

    print('search range:%d, %d' % (search_ymax - search_ymin + 1, search_xmax - search_xmin + 1))
    print('round 1:***************************' + str(time.time() - start_time))

    '''round 2'''
    scale_ratio = 1
    image_left = cv2.resize(image_left_ori, (int(image_left_ori.shape[1] / scale_ratio),
                                             int(image_left_ori.shape[0] / scale_ratio)), cv2.INTER_CUBIC)
    image_right = cv2.resize(image_right_ori, (int(image_right_ori.shape[1] / scale_ratio),
                                               int(image_right_ori.shape[0] / scale_ratio)), cv2.INTER_CUBIC)

    if plot is True:
        plt.imshow(image_left, cmap='gray')
        plt.show()
        plt.imshow(image_right, cmap='gray')
        plt.show()

    kx = kx_ori / scale_ratio
    ky = ky_ori / scale_ratio

    search_xmin = -offset_x - max(4, int(offset_x / 8))
    search_xmax = -offset_x + max(4, int(offset_x / 8))
    search_ymin = offset_y - 3
    search_ymax = offset_y + 3

    # print(search_xmin, search_xmax, search_ymin, search_ymax)

    window_all = np.array([], dtype='int32')
    target_all = np.array([], dtype='int32')
    window_width = np.zeros((len(human) - 1), dtype='int32')
    window_height = np.zeros((len(human) - 1), dtype='int32')
    target_width = np.zeros((len(human) - 1), dtype='int32')
    target_height = np.zeros((len(human) - 1), dtype='int32')

    ssd = np.zeros((len(human) - 1, search_ymax - search_ymin + 1, search_xmax - search_xmin + 1), dtype='int32')
    idx = 0
    for key in sorted(human):
        if key != 0:
            ymin_f, xmin_f, ymax_f, xmax_f = human[key]  # TODO change the range of head and hip

            if key == 1: # head top
                ymin_f = ymin_f * 0.7 + ymax_f * 0.3
            if key in [9, 10]: # left hip or right hip
                xmin_f = xmin_f - (xmax_f - xmin_f) * 0.3
                xmax_f = xmax_f + (xmax_f - xmin_f) * 0.3

            ymin = max(int(ymin_f * ky), 0)
            xmin = max(int(xmin_f * kx), 0)
            ymax = min(int(ymax_f * ky), image_left.shape[0])
            xmax = min(int(xmax_f * kx), image_left.shape[1])

            window = image_left[ymin:ymax, xmin:xmax].astype(np.int32)
            # print(ymin, ymax, xmin, xmax)

            target = np.full((window.shape[0] + search_ymax - search_ymin,
                              window.shape[1] + search_xmax - search_xmin), 512, dtype='int32')
            target_ymin = max(ymin + search_ymin, 0)
            target_ymax = min(ymax + search_ymax, image_right.shape[0])
            target_xmin = max(xmin + search_xmin, 0)
            target_xmax = min(xmax + search_xmax, image_right.shape[1])
            target_from_right = image_right[target_ymin:target_ymax, target_xmin:target_xmax]
            clip_ymin = max(-(ymin + search_ymin), 0)
            clip_ymax = max(ymax + search_ymax - image_right.shape[0], 0)

            target[clip_ymin: target.shape[0] - clip_ymax, target.shape[1] - (target_xmax - target_xmin):] = \
                target_from_right

            window_all = np.append(window_all, window.flatten())
            target_all = np.append(target_all, target.flatten())
            window_width[idx] = window.shape[1]
            window_height[idx] = window.shape[0]
            target_width[idx] = target.shape[1]
            target_height[idx] = target.shape[0]

            points[idx, 0] = (xmin_f + xmax_f) * kx / 2
            points[idx, 1] = (ymin_f + ymax_f) * ky / 2
            idx = idx + 1

    print('cpu:' + str(time.time() - start_time))
    out = np.zeros_like(ssd, dtype='int32')
    compute_ssd(window_all, target_all, out, window_width, window_height,
                target_width, target_height, ssd.shape[2], ssd.shape[1], len(human) - 1)
    ssd = out
    print('cuda ssd:' + str(time.time() - start_time))

    heat_map = ssd.sum(axis=0)

    offset_window = 0
    offset_target = 0
    if plot is True:
        for i in range(len(human) - 1):
            plt.subplot(311)
            plt.imshow(window_all[offset_window: offset_window + window_width[i] * window_height[i]]
                       .reshape((window_height[i], window_width[i])), cmap='gray')
            plt.subplot(312)
            plt.imshow(target_all[offset_target: offset_target + target_width[i] * target_height[i]]
                       .reshape((target_height[i], target_width[i])), cmap='gray')
            plt.subplot(313)
            plt.imshow(ssd[i], cmap='gray')
            plt.show()
            offset_window += window_width[i] * window_height[i]
            offset_target += target_width[i] * target_height[i]
        plt.imshow(heat_map, cmap='gray')
        plt.show()

    print('search range:%d, %d' % (search_ymax - search_ymin + 1, search_xmax - search_xmin + 1))
    print('round 2:+++++++++++++++++++++++++++' + str(time.time() - start_time))

    mask = np.ones((mask_size, mask_size), dtype='int32')
    convolved = signal.convolve2d(heat_map, mask, 'valid')
    min_idx = np.argmin(convolved.reshape(-1))
    center_x, center_y = min_idx % convolved.shape[1] + 1, min_idx / convolved.shape[1] + 1
    # print(center_x, center_y)

    offset_x = -(search_xmax - search_xmin - center_x) * scale_ratio + search_xmax
    offset_y = (search_ymin + center_y) * scale_ratio + (search_ymin + search_ymax) / 2

    candidate_range_x = max(3, int(math.fabs(offset_x / 16)))  # pm
    candidate_range_y = max(1, int(math.fabs(offset_y / 16)))  # pm

    '''print(offset_x)
    print(offset_y)
    print(candidate_range_x)
    print(candidate_range_y)'''
    print('offset: %d, %d' % (offset_y, offset_x))
    print('candidate_range:%d, %d' % (2 * candidate_range_y + 1, 2 * candidate_range_x + 1))

    for i in range(ssd.shape[0]): # TODO weighted interpolation
        candidate_ymin = max(center_y - candidate_range_y, 0)
        candidate_ymax = min(center_y + candidate_range_y + 1, ssd.shape[1])
        candidate_xmin = max(center_x - candidate_range_x, 0)
        candidate_xmax = min(center_x + candidate_range_x + 1, ssd.shape[2])
        candidate_rigion = ssd[i, candidate_ymin: candidate_ymax, candidate_xmin: candidate_xmax]
        if plot is True:
            plt.imshow(candidate_rigion, cmap='gray')
            plt.show()
        best_candidate_x = np.argmin(candidate_rigion.reshape(-1)) % (candidate_xmax - candidate_xmin)
        best_candidate_y = np.argmin(candidate_rigion.reshape(-1)) / (candidate_xmax - candidate_xmin)
        neighbor_xmin = max(best_candidate_x - 1, 0)
        neighbor_xmax = min(best_candidate_x + 2, candidate_rigion.shape[1])
        neighbor_ymin = max(best_candidate_y - 1, 0)
        neighbor_ymax = min(best_candidate_y + 2, candidate_rigion.shape[0])
        min_neighbor = candidate_rigion[neighbor_ymin: neighbor_ymax, neighbor_xmin: neighbor_xmax]
        min_neighbor = np.interp(min_neighbor, (np.min(min_neighbor.reshape(-1)), np.max(min_neighbor.reshape(-1))),
                                 interp_range)
        weight = 1.0 / min_neighbor
        norm_factor = np.sum(weight.reshape(-1))
        weight = weight / norm_factor
        disparity_center = candidate_range_x - best_candidate_x - offset_x
        disparity_left = disparity_center + 1 if neighbor_xmin == best_candidate_x - 1 else disparity_center
        disparity_right = disparity_left - weight.shape[1] + 1
        disparity_mat = np.repeat(np.linspace(disparity_left, disparity_right, num=weight.shape[1])[:, np.newaxis],
                                  weight.shape[0], axis=1).transpose()
        # print(disparity_mat)
        disparity = np.sum(np.multiply(weight, disparity_mat).reshape(-1))
        # print(disparity)
        depth = (FocalLength * BaseLine) / (disparity + 0.0000001)
        # print(depth)
        points[i, 2] = depth
    print(time.time() - start_time)
    for i in range(len(points)):
        points[i] = pix2cam(points[i])
    return points


def main():
    config = configparser.ConfigParser()
    config.read('config.ini', 'UTF-8')

    model = create_model(config)

    image_left_ori = cv2.imread('../images/img3_l.png')
    image_right_ori = cv2.imread('../images/img3_r.png')
    shape_ori = image_left_ori.shape
    image_left = cv2.cvtColor(image_left_ori, cv2.COLOR_BGR2RGB)
    image_left = cv2.resize(image_left, model.insize)
    with chainer.using_config('autotune', True):
        humans, _ = estimate(model, image_left.transpose(2, 0, 1).astype(np.float32))
        print(humans)
    pilImg = Image.fromarray(image_left)
    pilImg = draw_humans(
        model.keypoint_names,
        model.edges,
        pilImg,
        humans,
        mask=None
    )
    img_with_humans = cv2.cvtColor(np.asarray(pilImg), cv2.COLOR_RGB2BGR)
    if (len(humans)) > 0:
        shape_human = img_with_humans.shape
        kx = shape_ori[1] * 1.0 / shape_human[1]
        ky = shape_ori[0] * 1.0 / shape_human[0]
        points = get_points(humans[0], image_left_ori, image_right_ori, kx, ky)
    print(points)

    # img_with_humans = cv2.resize(img_with_humans, (shape_ori[1], shape_ori[0]))
    plt.figure()
    plt.imshow(img_with_humans)
    plt.show()


if __name__ == '__main__':
    main()
