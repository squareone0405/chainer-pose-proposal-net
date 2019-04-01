import matplotlib.pyplot as plt

from scipy import signal

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
    FocalLength = 350
    start_time = time.time()
    image_left = cv2.cvtColor(image_left, cv2.COLOR_RGB2GRAY)
    image_right = cv2.cvtColor(image_right, cv2.COLOR_RGB2GRAY)
    points = np.zeros((len(human) - 1, 3), dtype='float64')
    search_range_x = 30
    search_range_y = 3 # pm
    offset_y = 0
    mask_size = 3
    candidate_range_x = 3 # pm
    candidate_range_y = 1 # pm
    interp_range = (0.1, 1)
    ssd = np.zeros((len(human) - 1, search_range_y * 2 + 1, search_range_x), dtype='int32')
    idx = 0
    for key in sorted(human):
        if key != 0:
            ymin_f, xmin_f, ymax_f, xmax_f = human[key] # TODO range check
            ymin = int(ymin_f * ky)
            xmin = int(xmin_f * kx)
            ymax = int(ymax_f * ky)
            xmax = int(xmax_f * kx)
            window = image_left[ymin:ymax, xmin:xmax]
            target = image_right[offset_y + ymin - search_range_y:offset_y + ymax + search_range_y + 1,
                     xmin - search_range_x + 1:xmax]
            for i in range(search_range_y * 2 + 1):
                for j in range(search_range_x):
                    diff = target[i: i + window.shape[0], j: j + window.shape[1]] - window
                    ssd[idx, i, j] = np.sum(np.multiply(diff, diff).reshape(-1))

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
    heat_map = ssd.sum(axis=0)
    '''plt.imshow(heat_map, cmap='gray')
    plt.show()'''
    mask = np.ones((mask_size, mask_size), dtype='int32')
    convolved = signal.convolve2d(heat_map, mask, 'valid')
    min_idx = np.argmin(convolved.reshape(-1))
    print(min_idx)
    center_x, center_y = min_idx % convolved.shape[1] + 1, min_idx / convolved.shape[1] + 1
    print(center_x, center_y)
    '''weight = heat_map[center_y: center_y + mask_size, center_x: center_x + mask_size].astype('float64')
    norm_factor = np.sum(weight.reshape(-1))
    weight = weight / norm_factor'''

    for i in range(ssd.shape[0]): # TODO weighted interpolation
        candidate_rigion = ssd[i, center_y - candidate_range_y: center_y + candidate_range_y + 1,
                           center_x - candidate_range_x: center_x + candidate_range_x + 1]
        '''plt.imshow(candidate_rigion, cmap='gray')
        plt.show()'''
        best_candidate_x = (np.argmin(candidate_rigion.reshape(-1)) % (candidate_range_x * 2 + 1))
        best_candidate_y = (np.argmin(candidate_rigion.reshape(-1)) / (candidate_range_x * 2 + 1))
        print(best_candidate_x)
        neignbor_xmin = best_candidate_x - 1 if best_candidate_x -1 > 0 else 0
        neignbor_xmax = best_candidate_x + 2 if best_candidate_x + 2 < candidate_rigion.shape[1] \
            else candidate_rigion.shape[1]
        neignbor_ymin = best_candidate_y - 1 if best_candidate_y - 1 > 0 else 0
        neignbor_ymax = best_candidate_y + 2 if best_candidate_y + 2 < candidate_rigion.shape[0] \
            else candidate_rigion.shape[0]
        min_neignbor = candidate_rigion[neignbor_ymin: neignbor_ymax, neignbor_xmin: neignbor_xmax]
        min_neignbor = np.interp(min_neignbor, (np.min(min_neignbor.reshape(-1)), np.max(min_neignbor.reshape(-1))),
                                 interp_range)
        weight = 1.0 / min_neignbor
        norm_factor = np.sum(weight.reshape(-1))
        weight = weight / norm_factor
        print(weight)
        disparity_center = search_range_x - 1 - (center_x - candidate_range_x + best_candidate_x)
        print(disparity_center)
        disparity_left = disparity_center + 1 if disparity_center + 1 < search_range_x else disparity_center
        print(disparity_left)
        disparity_right = disparity_left - weight.shape[1] + 1 if disparity_left - weight.shape[1] + 1 >= 0 else 0
        print(disparity_right)
        disparity_mat = np.repeat(np.linspace(disparity_left, disparity_right, num=weight.shape[1])[:, np.newaxis],
                                  weight.shape[0], axis=1).transpose()
        print(disparity_mat)
        disparity = np.sum(np.multiply(weight, disparity_mat).reshape(-1))
        # disparity = search_range_x - 1 - (center_x - candidate_range_x + best_candidate_x)
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

    image_left_ori = cv2.imread('lwb_l.png')
    image_right_ori = cv2.imread('lwb_r.png')
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
