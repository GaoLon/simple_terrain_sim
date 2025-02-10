cmake_minimum_required(VERSION 2.8.3)
project(kimatic_simulator)

set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_CXX_FLAGS "-std=c++11")
ADD_COMPILE_OPTIONS(-std=c++11 )
ADD_COMPILE_OPTIONS(-std=c++14 )
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  nav_msgs
  std_msgs
  tf
  roslib
  pcl_conversions
  visualization_msgs
)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

add_message_files(
  FILES
  DiffCMD.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
 CATKIN_DEPENDS roscpp std_msgs tf visualization_msgs
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

add_executable (kimatic_simulator_node src/poscmd_2_odom.cpp )
add_dependencies(kimatic_simulator_node kimatic_simulator_gencpp)
target_link_libraries(kimatic_simulator_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)
