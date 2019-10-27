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

QUEUE_SIZE = 5

"""
Bonus script
If you have good USB camera which gets image as well as 60 FPS,
this script will be helpful for realtime inference
"""


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

    def run(self):
        while not self.stop_event.is_set():
            try:
                image = self.cap.get()
                hog = cv2.HOGDescriptor()
                hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                boxes, weights = hog.detectMultiScale(image, winStride=(8, 8))
                boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
                feature_maps = []
                with chainer.using_config('autotune', True), \
                        chainer.using_config('use_ideep', 'auto'):
                    for box in boxes:
                        roi = cv2.resize(image[box[1]: box[3], box[0]: box[2]], self.cap.insize)
                        # cv2.imshow('roi', roi)
                        feature_maps.append(get_feature(self.model, roi.transpose(2, 0, 1).astype(np.float32)))
                if len(boxes):
                    self.queue.put((image, feature_maps, boxes, weights), timeout=1)
            except Queue.Full:
                pass
            except Queue.Empty:
                pass

    def get(self):
        return self.queue.get(timeout=1)

    def stop(self):
        logger.info('{} will stop'.format(self.name))
        self.stop_event.set()


class Tracker:
    def __init__(self):
        self.centers = np.zeros((0, 2))
        self.centers_ttl = np.array([], dtype=np.int32)
        self.cv_trackers = np.array([])
        self.cv_trackers_ttl = np.array([], dtype=np.int32)
        self.cv_trackers_life = 30
        self.dist_thresh_detect = 20
        self.dist_thresh_tracking = 15
        self.center_life = 40
        self.iou_thresh = 0.4

    def update(self, detect_boxes, image):  # tracking box and detect box both in xy(width, height) order
        print("in update------------------------------------------------------")
        tracking_centers = np.zeros((0, 2))
        tracking_boxes = np.zeros((0, 4))
        detect_centers = np.zeros((0, 2))
        cv_tracker_mask = np.array([True] * self.cv_trackers.shape[0])
        for i in range(self.cv_trackers.shape[0]):
            ok, tracking_box = self.cv_trackers[i].update(image)
            if ok:
                p1 = (int(tracking_box[0]), int(tracking_box[1]))
                p2 = (int(tracking_box[0] + tracking_box[2]), int(tracking_box[1] + tracking_box[3]))
                cv2.rectangle(image, p1, p2, (255, 0, 0), 2, 1)
                tracking_boxes = np.append(tracking_boxes, np.array([[p1[0], p1[1], p2[0], p2[1]]]), axis=0)
                tracking_centers = np.append(tracking_centers, np.array([[(p1[0] + p2[0]) / 2.0,
                                                                          (p1[1] + p2[1]) / 2.0]]), axis=0)
            else:
                cv_tracker_mask[i] = False
                cv2.putText(image, "Tracking failure detected", (100, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        if self.cv_trackers.shape[0]:
            self.cv_trackers = self.cv_trackers[cv_tracker_mask]
            self.cv_trackers_ttl = self.cv_trackers_ttl[cv_tracker_mask]

        if not self.centers.shape[0]:
            for detect_box in detect_boxes:
                tracker = cv2.TrackerKCF_create()
                ok = tracker.init(image, (detect_box[0], detect_box[1], detect_box[2] - detect_box[0],
                                          detect_box[3] - detect_box[1]))
                if ok:
                    self.cv_trackers = np.append(self.cv_trackers, tracker)
                    self.cv_trackers_ttl = np.append(self.cv_trackers_ttl, self.cv_trackers_life)
                self.centers = np.append(self.centers, np.array([[(detect_box[0] + detect_box[2]) / 2.0,
                                                                  (detect_box[1] + detect_box[3]) / 2.0]]), axis=0)
                self.centers_ttl = np.append(self.centers_ttl, self.center_life)
        else:
            for detect_box in detect_boxes:
                detect_centers = np.append(detect_centers, np.array([[(detect_box[0] + detect_box[2]) / 2.0,
                                                                      (detect_box[1] + detect_box[3]) / 2.0]]), axis=0)
                cv2.rectangle(image, (int(detect_box[0]), int(detect_box[1])),
                              (int(detect_box[2]), int(detect_box[3])), (0, 255, 0), 2, 1)
            dist_detect = np.zeros((self.centers.shape[0], detect_centers.shape[0]))
            for i in range(self.centers.shape[0]):
                for j in range(detect_centers.shape[0]):
                    dist_detect[i][j] = self.get_dist(self.centers[i, :], detect_centers[j, :])
            dist_tracking = np.zeros((self.centers.shape[0], tracking_boxes.shape[0]))
            for i in range(self.centers.shape[0]):
                for j in range(tracking_boxes.shape[0]):
                    dist_tracking[i][j] = self.get_dist(self.centers[i, :], tracking_centers[j, :])
            print("tracking boxes")
            print(tracking_boxes)
            print("tracking centers")
            print(tracking_centers)
            print("detect boxes")
            print(detect_boxes)
            print("detect centers")
            print(detect_centers)
            print("dist detect")
            print(dist_detect)
            print("dist tracking")
            print(dist_tracking)
            print("centers")
            print(self.centers)
            print("ttl")
            print(self.centers_ttl)
            print("******************************")
            detect_used = np.array([False] * detect_boxes.shape[0], dtype=np.bool)
            tracking_used = np.array([False] * tracking_boxes.shape[0], dtype=np.bool)
            print(tracking_used)

            re_track = []

            for i in range(self.centers.shape[0]):
                weight_last = 1.0
                weight_detect = 0.0
                weight_tracking = 0.0
                detect_center = np.zeros((1, 2))
                tracking_center = np.zeros((1, 2))
                min_detect_idx = 0
                if dist_detect.shape[1]:
                    min_detect_idx = np.argmin(dist_detect[i, :])
                    if dist_detect[i, min_detect_idx] < self.dist_thresh_detect:
                        weight_detect = 1.0
                        detect_center = detect_centers[min_detect_idx]
                        detect_used[min_detect_idx] = True
                if dist_tracking.shape[1]:
                    min_tracking_idx = np.argmin(dist_tracking[i, :])
                    if dist_tracking[i, min_tracking_idx] < self.dist_thresh_tracking:
                        weight_tracking = 1.0
                        tracking_center = tracking_centers[min_tracking_idx]
                        tracking_used[min_tracking_idx] = True
                self.centers[i] = (self.centers[i] * weight_last + detect_center * weight_detect +
                                   tracking_center * weight_tracking) / (weight_last + weight_detect + weight_tracking)
                if (weight_detect + weight_tracking) == 0.0:
                    self.centers_ttl[i] -= 2
                if weight_tracking == 1.0 and weight_detect == 0.0:
                    self.centers_ttl[i] -= 1
                if weight_tracking == 0.0 and weight_detect == 1.0:
                    re_track.append(min_detect_idx)
                if (weight_detect + weight_tracking) == 2.0:
                    self.centers_ttl[i] = self.center_life

            center_mask = [True] * self.centers.shape[0]
            for i in range(self.centers.shape[0]):
                if self.centers_ttl[i] <= 0:
                    center_mask[i] = False
            self.centers = self.centers[center_mask, :]
            self.centers_ttl = self.centers_ttl[center_mask]

            if np.sum(tracking_used) < self.cv_trackers.shape[0]:
                print(tracking_used)
                print(self.cv_trackers_ttl)
                print(self.cv_trackers_ttl[~tracking_used])
                self.cv_trackers_ttl[~tracking_used] = self.cv_trackers_ttl[~tracking_used] - 1
                cv_tracker_mask = self.cv_trackers_ttl > 0
                print(cv_tracker_mask)
                self.cv_trackers = self.cv_trackers[cv_tracker_mask]
                self.cv_trackers_ttl = self.cv_trackers_ttl[cv_tracker_mask]

            for detect_idx in re_track:
                detect_box = detect_boxes[detect_idx]
                tracker = cv2.TrackerKCF_create()
                ok = tracker.init(image, (detect_box[0], detect_box[1], detect_box[2] - detect_box[0],
                                          detect_box[3] - detect_box[1]))
                if ok:
                    self.cv_trackers = np.append(self.cv_trackers, tracker)
                    self.cv_trackers_ttl = np.append(self.cv_trackers_ttl, self.cv_trackers_life)
                self.centers = np.append(self.centers, np.array([[(detect_box[0] + detect_box[2]) / 2.0,
                                                                  (detect_box[1] + detect_box[3]) / 2.0]]), axis=0)
                self.centers_ttl = np.append(self.centers_ttl, self.center_life)

            if np.sum(detect_used) < detect_boxes.shape[0]:
                for detect_box in detect_boxes[~detect_used]:
                    new_box = True
                    for tracking_box in tracking_boxes:
                        if self.get_iou(tracking_box, detect_box) > self.iou_thresh:
                            new_box = False
                            break
                    if new_box:
                        tracker = cv2.TrackerKCF_create()
                        ok = tracker.init(image, (detect_box[0], detect_box[1], detect_box[2] - detect_box[0],
                                                  detect_box[3] - detect_box[1]))
                        if ok:
                            self.cv_trackers = np.append(self.cv_trackers, tracker)
                            self.cv_trackers_ttl = np.append(self.cv_trackers_ttl, self.cv_trackers_life)
                        self.centers = np.append(self.centers, np.array([[(detect_box[0] + detect_box[2]) / 2.0,
                                                                          (detect_box[1] + detect_box[3]) / 2.0]]), axis=0)
                        self.centers_ttl = np.append(self.centers_ttl, self.center_life)

        duplicate_mask = [True] * self.centers.shape[0]
        for i in range(self.centers.shape[0]):
            for j in np.arange(i + 1, self.centers.shape[0]):
                if self.get_dist(self.centers[i], self.centers[j]) < 1:
                    duplicate_mask[j] = False
        self.centers = self.centers[duplicate_mask, :]
        self.centers_ttl = self.centers_ttl[duplicate_mask]

        for i in range(self.centers.shape[0]):
            cv2.putText(image, "%i" % self.centers_ttl[i], (int(self.centers[i, 0]), int(self.centers[i, 1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.rectangle(image, (int(self.centers[i, 0] - 2), int(self.centers[i, 1] - 2)),
                          (int(self.centers[i, 0] + 2), int(self.centers[i, 1] + 2)), (255, 255, 0), 2, 2)
        cv2.putText(image, "center num: %i, tracking: %i, detect: %i" % (self.centers.shape[0],
                                                                         tracking_boxes.shape[0],
                                                                         detect_boxes.shape[0]), (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    def get_dist(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) * (point1[0] - point2[0]) +
                         (point1[1] - point2[1]) * (point1[1] - point2[1]))

    def get_iou(self, box1, box2):
        left = max(box1[0], box2[0])
        top = max(box1[1], box2[1])
        right = min(box1[2], box2[2])
        bottom = min(box1[3], box2[3])
        intersection = (right - left) * (bottom - top)
        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou = 0.0
        if right < left or bottom < top:
            iou = 0.0
        else:
            iou = intersection / (area_box1 + area_box2 - intersection)
        print("iou: %f" % iou)
        return iou


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

    kx = 1.0 * 672 / 224
    ky = 1.0 * 376 / 224
    # init = False

    tracker = Tracker()

    try:
        while not main_event.is_set() and cap.isOpened():
            degree += 5
            degree = degree % 360
            try:
                image, feature_maps, boxes, weights = predictor.get()
                humans = []
                confidences = []
                for feature_map in feature_maps:
                    h, c = get_humans_by_feature(model, feature_map)
                    if len(h) == 1:
                        humans.append(h)
                        confidences.append(c)
                    # humans, confidences = get_humans_by_feature(model, feature_map)
                print('*******************')
                print(humans)
                print(confidences)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                bbox_heads = np.zeros((len(humans), 4))
                for i in range(len(humans)):
                    bbox_head_ = humans[i][0][0]
                    # print("origin")
                    # print(bbox_head_)
                    bbox_head = copy.deepcopy(bbox_head_)
                    detect_height = boxes[i][3] - boxes[i][1]
                    detect_width = boxes[i][2] - boxes[i][0]
                    bbox_head[0] = bbox_head[0] * detect_height / 224 + boxes[i][1]
                    bbox_head[1] = bbox_head[1] * detect_width / 224 + boxes[i][0]
                    bbox_head[2] = bbox_head[2] * detect_height / 224 + boxes[i][1]
                    bbox_head[3] = bbox_head[3] * detect_width / 224 + boxes[i][0]
                    bbox_head_width = bbox_head[3] - bbox_head[1]
                    bbox_head_height = bbox_head[2] - bbox_head[0]
                    bbox_head[1] += 0.2 * bbox_head_width
                    bbox_head[3] -= 0.2 * bbox_head_width
                    bbox_head[0] += 0.0 * bbox_head_height
                    bbox_head[2] += 0.1 * bbox_head_height
                    # yx order to xy order
                    bbox_heads[i, :] = np.array([bbox_head[1], bbox_head[0], bbox_head[3], bbox_head[2]])

                for i in range(len(humans)):
                    detect_height = boxes[i][3] - boxes[i][1]
                    detect_width = boxes[i][2] - boxes[i][0]
                    for t in humans[i][0]:
                        bbox_ = humans[i][0][t]
                        bbox_[0] = (bbox_[0] * detect_height / 224 + boxes[i][1]) / ky
                        bbox_[1] = (bbox_[1] * detect_width / 224 + boxes[i][0]) / kx
                        bbox_[2] = (bbox_[2] * detect_height / 224 + boxes[i][1]) / ky
                        bbox_[3] = (bbox_[3] * detect_width / 224 + boxes[i][0]) / kx

                tracker.update(bbox_heads, image)
                for (xA, yA, xB, yB) in boxes:
                    # display the detected boxes in the colour picture
                    cv2.rectangle(image, (xA, yA), (xB, yB), (255, 255, 0), 2)
                    print('++++++++++++++++++++')
                    print(boxes)
                cv2.imshow('tracking', image)
                image = cv2.resize(image, (224, 224))
            except Queue.Empty:
                continue
            except Exception:
                break
            pilImg = Image.fromarray(image)
            for human in humans:
                pilImg = draw_humans(
                    model.keypoint_names,
                    model.edges,
                    pilImg,
                    human,
                    mask=mask.rotate(degree) if mask else None
                )
            # img_with_humans = cv2.cvtColor(np.asarray(pilImg), cv2.COLOR_RGB2BGR)
            img_with_humans = np.array(pilImg)
            msg = 'GPU ON' if chainer.backends.cuda.available else 'GPU OFF'
            msg += ' ' + config.get('model_param', 'model_name')
            cv2.putText(img_with_humans, 'FPS: %f' % (1.0 / (time.time() - fps_time)),
                        (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            img_with_humans = cv2.resize(img_with_humans, (672, 376))
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
    main()
