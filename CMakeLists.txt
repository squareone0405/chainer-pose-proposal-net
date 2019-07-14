cmake_minimum_required(VERSION 2.8.3)

project(ppn)

set(PACKAGE_DEPENDENCIES
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
)

find_package(catkin REQUIRED COMPONENTS
  ${PACKAGE_DEPENDENCIES}
  message_generation
)

add_message_files(
  FILES
  Skeleton.msg
  Bbox.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
)

catkin_package(
  CATKIN_DEPENDS
  ${PACKAGE_DEPENDENCIES}
  message_runtime
)

