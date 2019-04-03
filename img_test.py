import matplotlib.pyplot as plt

from scipy import signal
import math

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


def get_points(human, image_left, image_right, kx, ky):
    BaseLine = 0.12
    FocalLength = 1400

    points = np.zeros((len(human) - 1, 3), dtype='float64')
    mask_size = 3
    interp_range = (0.1, 1)

    start_time = time.time()
    image_left_ori = cv2.cvtColor(image_left, cv2.COLOR_RGB2GRAY)
    image_right_ori = cv2.cvtColor(image_right, cv2.COLOR_RGB2GRAY)
    kx_ori = kx
    ky_ori = ky

    scale_ratio = 8
    image_left = cv2.resize(image_left_ori, (int(image_left_ori.shape[1] / scale_ratio),
                                         int(image_left_ori.shape[0] / scale_ratio)), cv2.INTER_CUBIC)
    image_right = cv2.resize(image_right_ori, (int(image_right_ori.shape[1] / scale_ratio),
                                         int(image_right_ori.shape[0] / scale_ratio)), cv2.INTER_CUBIC)

    '''plt.imshow(image_left, cmap='gray')
    plt.show()
    plt.imshow(image_right, cmap='gray')
    plt.show()'''

    kx = kx_ori / scale_ratio
    ky = ky_ori / scale_ratio

    search_xmin = -20
    search_xmax = 0
    search_ymin = -3
    search_ymax = 3

    ssd = np.zeros((len(human) - 1, search_ymax - search_ymin + 1, search_xmax - search_xmin + 1), dtype='int32')
    idx = 0
    ticx = time.time()
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

            window = image_left[ymin:ymax, xmin:xmax]
            print(ymin, ymax, xmin, xmax)

            target = np.full((window.shape[0] + search_ymax - search_ymin,
                              window.shape[1] + search_xmax - search_xmin), 512, dtype='int16')
            target_ymin = max(ymin + search_ymin, 0)
            target_ymax = min(ymax + search_ymax, image_right.shape[0])
            target_xmin = max(xmin + search_xmin, 0)
            target_xmax = min(xmax + search_xmax, image_right.shape[1])
            target_from_right = image_right[target_ymin:target_ymax, target_xmin:target_xmax]
            clip_ymin = max(-(ymin + search_ymin), 0)
            clip_ymax = max(ymax + search_ymax - image_right.shape[0], 0)

            target[clip_ymin: target.shape[0] - clip_ymax, target.shape[1] - (target_xmax - target_xmin):] = \
                target_from_right

            tic = time.time()
            for i in range(search_ymax - search_ymin + 1):
                for j in range(search_xmax - search_xmin + 1):
                    diff = target[i: i + window.shape[0], j: j + window.shape[1]] - window
                    ssd[idx, i, j] = np.sum(np.multiply(diff, diff).reshape(-1))
            print('ssd time++++++++++++++++++++++++++++++++++++')
            print(time.time() - tic)

            '''plt.subplot(311)
            plt.imshow(window, cmap='gray')
            plt.subplot(312)
            plt.imshow(target, cmap='gray')
            plt.subplot(313)
            plt.imshow(ssd[idx], cmap='gray')
            plt.show()'''

            points[idx, 0] = (xmin_f + xmax_f) * kx / 2
            points[idx, 1] = (ymin_f + ymax_f) * ky / 2
            idx = idx + 1
    print(time.time() - ticx)
    heat_map = ssd.sum(axis=0)
    '''plt.imshow(heat_map, cmap='gray')
    plt.show()'''
    mask = np.ones((mask_size, mask_size), dtype='int32')
    convolved = signal.convolve2d(heat_map, mask, 'valid')
    min_idx = np.argmin(convolved.reshape(-1))
    print(min_idx)
    center_x, center_y = min_idx % convolved.shape[1] + 1, min_idx / convolved.shape[1] + 1
    print(center_x, center_y)

    offset_x = (search_xmax - search_xmin - center_x) * scale_ratio
    offset_y = (search_ymin + center_y) * scale_ratio

    '''round 2'''
    scale_ratio = 1
    image_left = cv2.resize(image_left_ori, (int(image_left_ori.shape[1] / scale_ratio),
                                             int(image_left_ori.shape[0] / scale_ratio)), cv2.INTER_CUBIC)
    image_right = cv2.resize(image_right_ori, (int(image_right_ori.shape[1] / scale_ratio),
                                               int(image_right_ori.shape[0] / scale_ratio)), cv2.INTER_CUBIC)

    kx = kx_ori / scale_ratio
    ky = ky_ori / scale_ratio

    search_xmin = -offset_x - max(4, int(offset_x / 6))
    search_xmax = -offset_x + max(4, int(offset_x / 6))
    search_ymin = offset_y - 3
    search_ymax = offset_y + 3

    print(search_xmin, search_xmax, search_ymin, search_ymax)

    ssd = np.zeros((len(human) - 1, search_ymax - search_ymin + 1, search_xmax - search_xmin + 1), dtype='int32')
    idx = 0
    ticx = time.time()
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

            window = image_left[ymin:ymax, xmin:xmax]
            print(ymin, ymax, xmin, xmax)

            target = np.full((window.shape[0] + search_ymax - search_ymin,
                              window.shape[1] + search_xmax - search_xmin), 512, dtype='int16')
            target_ymin = max(ymin + search_ymin, 0)
            target_ymax = min(ymax + search_ymax, image_right.shape[0])
            target_xmin = max(xmin + search_xmin, 0)
            target_xmax = min(xmax + search_xmax, image_right.shape[1])
            target_from_right = image_right[target_ymin:target_ymax, target_xmin:target_xmax]
            clip_ymin = max(-(ymin + search_ymin), 0)
            clip_ymax = max(ymax + search_ymax - image_right.shape[0], 0)

            target[clip_ymin: target.shape[0] - clip_ymax, target.shape[1] - (target_xmax - target_xmin):] = \
                target_from_right

            tic = time.time()
            for i in range(search_ymax - search_ymin + 1):
                for j in range(search_xmax - search_xmin + 1):
                    diff = target[i: i + window.shape[0], j: j + window.shape[1]] - window
                    ssd[idx, i, j] = np.sum(np.multiply(diff, diff).reshape(-1))
            print('ssd time---------------------------------')
            print(time.time() - tic)

            '''plt.subplot(311)
            plt.imshow(window, cmap='gray')
            plt.subplot(312)
            plt.imshow(target, cmap='gray')
            plt.subplot(313)
            plt.imshow(ssd[idx], cmap='gray')
            plt.show()'''

            points[idx, 0] = (xmin_f + xmax_f) * kx / 2
            points[idx, 1] = (ymin_f + ymax_f) * ky / 2
            idx = idx + 1
    print(time.time() - ticx)
    heat_map = ssd.sum(axis=0)
    '''plt.imshow(heat_map, cmap='gray')
    plt.show()'''
    mask = np.ones((mask_size, mask_size), dtype='int32')
    convolved = signal.convolve2d(heat_map, mask, 'valid')
    min_idx = np.argmin(convolved.reshape(-1))
    print(min_idx)
    center_x, center_y = min_idx % convolved.shape[1] + 1, min_idx / convolved.shape[1] + 1
    print(center_x, center_y)

    offset_x = -(search_xmax - search_xmin - center_x) * scale_ratio + search_xmax
    offset_y = (search_ymin + center_y) * scale_ratio + (search_ymin + search_ymax) / 2

    candidate_range_x = max(3, int(math.fabs(offset_x / 8)))  # pm
    candidate_range_y = max(1, int(math.fabs(offset_y / 8)))  # pm

    print(offset_x)
    print(offset_y)
    print(candidate_range_x)
    print(candidate_range_y)

    for i in range(ssd.shape[0]): # TODO weighted interpolation
        candidate_ymin = max(center_y - candidate_range_y, 0)
        candidate_ymax = min(center_y + candidate_range_y + 1, ssd.shape[1])
        candidate_xmin = max(center_x - candidate_range_x, 0)
        candidate_xmax = min(center_x + candidate_range_x + 1, ssd.shape[2])
        candidate_rigion = ssd[i, candidate_ymin: candidate_ymax, candidate_xmin: candidate_xmax]
        '''plt.imshow(candidate_rigion, cmap='gray')
        plt.show()'''
        best_candidate_x = (np.argmin(candidate_rigion.reshape(-1)) % (candidate_range_x * 2 + 1))
        best_candidate_y = (np.argmin(candidate_rigion.reshape(-1)) / (candidate_range_x * 2 + 1))
        # print(best_candidate_x)
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
        print(weight)
        # disparity_center = search_xmax - search_xmin - (center_x - candidate_range_x + best_candidate_x) - offset_x
        disparity_center = candidate_range_x - best_candidate_x - offset_x
        disparity_left = disparity_center + 1 if neighbor_xmin == best_candidate_x - 1 else disparity_center
        disparity_right = disparity_left - weight.shape[1] + 1
        disparity_mat = np.repeat(np.linspace(disparity_left, disparity_right, num=weight.shape[1])[:, np.newaxis],
                                  weight.shape[0], axis=1).transpose()
        print(disparity_mat)
        disparity = np.sum(np.multiply(weight, disparity_mat).reshape(-1))
        # disparity = search_xmax - search_xmin - (center_x - candidate_range_x + best_candidate_x)
        print(disparity)
        depth = (FocalLength * BaseLine) / (disparity + 0.0000001)
        print(depth)
        points[i, 2] = depth
        print('----------------')
    print(time.time() - start_time) # TODO speed up with cuda
    return points


def main():
    config = configparser.ConfigParser()
    config.read('config.ini', 'UTF-8')

    model = create_model(config)

    image_left_ori = cv2.imread('czp_l.png')
    image_right_ori = cv2.imread('czp_r.png')
    shape_ori = image_left_ori.shape
    image_left = cv2.cvtColor(image_left_ori, cv2.COLOR_BGR2RGB)
    image_left = cv2.resize(image_left, model.insize)
    with chainer.using_config('autotune', True):
        humans = estimate(model,
                          image_left.transpose(2, 0, 1).astype(np.float32))
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
