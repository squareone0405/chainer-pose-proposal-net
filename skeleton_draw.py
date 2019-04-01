#!/usr/bin/python2

import numpy as np
import rospy
import std_msgs
from geometry_msgs.msg import Point
from ppn.msg import Skeleton
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

COLOR_LIST = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 127, 0),
    (0, 255, 85),
    (0, 170, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255)
]


def skeleton_callback(skeleton):
    if skeleton.human_num > 0:
        point_list = np.array([])
        msg = MarkerArray()
        msg.markers = [None] * (skeleton.human_num * 14)
        for human_idx in range(skeleton.human_num):
            for skeleton_idx in range(14):
                invisible_marker = Marker()
                invisible_marker.header = std_msgs.msg.Header()
                invisible_marker.header.frame_id = 'map'
                invisible_marker.header.stamp = rospy.Time.now()
                invisible_marker.id = skeleton_idx + 1
                msg.markers[human_idx * 14 + skeleton_idx] = invisible_marker
            print('////////////////////////')
            print(int(str(skeleton.skeleton_num[human_idx]).encode('hex'), 16))
            if int(str(skeleton.skeleton_num[human_idx]).encode('hex'), 16) < 4:
                print('skipppppppppppppppppppp')
                continue
            index = 0
            for skeleton_idx in range(ord(skeleton.skeleton_num[human_idx])):
                marker = Marker()
                marker.header = std_msgs.msg.Header()
                marker.header.frame_id = 'map'
                marker.header.stamp = rospy.Time.now()
                marker.id = ord(skeleton.types[index])
                marker.pose.position.x = skeleton.points[index].x / 100
                marker.pose.position.y = skeleton.points[index].y / 100
                marker.pose.position.z = skeleton.points[index].z / 1000
                # marker.pose.position.z = np.random.normal(loc=0.0, scale=0.1, size=1)[0]
                point_list = np.append(point_list,
                                       [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z], axis=0)
                '''print("x:" + str(marker.pose.position.x))
                print("y:" + str(marker.pose.position.y))
                print("z:" + str(marker.pose.position.z))
                print("id:" + str(marker.id))'''
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.a = 1.0
                marker.color.r = COLOR_LIST[ord(skeleton.types[index]) - 1][0] / 255.0
                marker.color.g = COLOR_LIST[ord(skeleton.types[index]) - 1][1] / 255.0
                marker.color.b = COLOR_LIST[ord(skeleton.types[index]) - 1][2] / 255.0
                index = index + 1
                msg.markers[human_idx * 14 + skeleton_idx] = marker
            point_list = point_list.reshape((-1, 3))
            center = np.average(point_list, axis=0)
            centered_point = point_list - np.mean(point_list, axis=0)
            U, S, V = np.linalg.svd(centered_point, full_matrices=False)
            print('mean')
            print(np.mean(point_list, axis=0))
            print('***************************P')
            print(centered_point)
            print('---------------------------U')
            print(S)
            print('+++++++++++++++++++++++++++V')
            print(V)
            for arrow_idx in range(3):
                marker = Marker()
                marker.header = std_msgs.msg.Header()
                marker.header.frame_id = 'map'
                marker.header.stamp = rospy.Time.now()
                marker.id = 15 + arrow_idx
                marker.type = Marker.LINE_LIST
                marker.action = Marker.ADD
                p_start = Point()
                p_start.x = center[0]
                p_start.y = center[1]
                p_start.z = center[2]
                p_end = Point()
                p_end.x = center[0] + V[arrow_idx, 0]
                p_end.y = center[1] + V[arrow_idx, 1]
                p_end.z = center[2] + V[arrow_idx, 2]
                marker.points.append(p_start)
                marker.points.append(p_end)
                marker.pose.position.x = 0
                marker.pose.position.y = 0
                marker.pose.position.z = 0
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.a = 1.0
                if arrow_idx == 0:
                    marker.color.r = 1.0
                elif arrow_idx == 1:
                    marker.color.g = 1.0
                elif arrow_idx == 2:
                    marker.color.b = 1.0
                msg.markers.append(marker)
        pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node('skeleton_draw', anonymous=True)
    pub = rospy.Publisher('skeleton_marker', MarkerArray, queue_size=1000)
    sub = rospy.Subscriber('skeleton', Skeleton, skeleton_callback)
    rospy.spin()

