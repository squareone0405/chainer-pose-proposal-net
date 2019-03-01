#!/usr/bin/python2

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
        msg = MarkerArray()
        msg.markers = [None] * (skeleton.human_num * 14)
        for i in range(skeleton.human_num):
            for j in range(14):
                invisible_marker = Marker()
                invisible_marker.header = std_msgs.msg.Header()
                invisible_marker.header.frame_id = 'map'
                invisible_marker.header.stamp = rospy.Time.now()
                invisible_marker.id = j + 1
                msg.markers[i * 14 + j] = invisible_marker
            index = 0
            for j in range(ord(skeleton.skeleton_num[i])):
                marker = Marker()
                marker.header = std_msgs.msg.Header()
                marker.header.frame_id = 'map'
                marker.header.stamp = rospy.Time.now()
                marker.id = ord(skeleton.types[index])
                marker.pose.position.x = skeleton.points[index].x / 100
                marker.pose.position.y = skeleton.points[index].y / 100
                marker.pose.position.z = 0
                print("x:" + str(marker.pose.position.x))
                print("y:" + str(marker.pose.position.y))
                print("z:" + str(marker.pose.position.z))
                print("id:" + str(marker.id))
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
                msg.markers[i * 14 + j] = marker
        print(len(msg.markers))
        pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('skeleton_draw', anonymous=True)
    pub = rospy.Publisher('skeleton_marker', MarkerArray, queue_size=1000)
    sub = rospy.Subscriber('skeleton', Skeleton, skeleton_callback)
    rospy.spin()

