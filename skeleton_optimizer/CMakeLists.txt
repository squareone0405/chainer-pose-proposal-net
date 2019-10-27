cmake_minimum_required(VERSION 3.10)
project(skeleton_optimizer)

set(CMAKE_CXX_STANDARD 11)

find_package( Sophus REQUIRED )
find_package( Ceres REQUIRED )
find_package( Eigen3 REQUIRED)
include_directories( ${CERES_INCLUDE_DIRS} ${Sophus_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIRS})
link_directories( ${CERES_LIBRARIES_DIRS} )

add_library(skeleton SHARED optimizer.cpp optimizer.h)
target_link_libraries(skeleton ${CERES_LIBRARIES} ${Sophus_LIBRARIES})

add_executable(skeleton_optimizer main.cpp optimizer.cpp optimizer.h)
target_link_libraries(skeleton_optimizer ${CERES_LIBRARIES} ${Sophus_LIBRARIES})
