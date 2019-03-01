import matplotlib.pyplot as plt
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


def main():
    config = configparser.ConfigParser()
    config.read('config.ini', 'UTF-8')

    model = create_model(config)

    if os.path.exists('mask.png'):
        mask = Image.open('mask.png')
        mask = mask.resize((200, 200))
    else:
        mask = None

    image = cv2.imread('hyc1_l.png')
    shape_ori = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, model.insize)
    with chainer.using_config('autotune', True):
        humans = estimate(model,
                          image.transpose(2, 0, 1).astype(np.float32))
    pilImg = Image.fromarray(image)
    pilImg = draw_humans(
        model.keypoint_names,
        model.edges,
        pilImg,
        humans,
    )
    img_with_humans = cv2.cvtColor(np.asarray(pilImg), cv2.COLOR_RGB2BGR)
    if (len(humans)) > 0:
        if len(humans[0]) > 8:
            print(humans)
            plt.imshow(img_with_humans)
            plt.show()
            f = open('1.csv', 'w')
            shape_human = img_with_humans.shape
            print(shape_ori)
            kx = shape_ori[1] * 1.0 / shape_human[1]
            ky = shape_ori[0] * 1.0 / shape_human[0]
            print(shape_human)
            print(kx)
            print(ky)
            for human in humans:
                for k, b in human.items():
                    f.write(str(k))
                    f.write(',')
                    ymin, xmin, ymax, xmax = b
                    f.write(str((xmin + xmax) * kx / 2))
                    f.write(',')
                    f.write(str((ymin + ymax) * ky / 2))
                    f.write('\r\n')

    '''img_with_humans = cv2.resize(img_with_humans, (3 * model.insize[0], 3 * model.insize[1]))
    plt.imshow(img_with_humans)
    plt.show()'''

if __name__ == '__main__':
    main()
