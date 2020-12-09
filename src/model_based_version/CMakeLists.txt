cmake_minimum_required(VERSION 2.8.3)
project(model_based_version)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  image_transport
  roscpp
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
  rosbag
  roslib
  message_generation
  tf
)

find_package(Eigen3 REQUIRED)

catkin_package(
  # INCLUDE_DIRS include
  # LIBRARIES deep_sort_pytorch_package
  CATKIN_DEPENDS nav_msgs roscpp rospy sensor_msgs std_msgs tf rosbag roslib message_generation message_runtime
)

include_directories(
        ${PROJECT_SOURCE_DIR}/include
)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(odom_map_model_based src/odom_map.cpp)
target_link_libraries(odom_map_model_based ${catkin_LIBRARIES})


