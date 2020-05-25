cmake_minimum_required(VERSION 2.8.3)
project(klt_ros)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  geometry_msgs
  image_transport
  cv_bridge
)

## System dependencies are found with CMake's conventions
 find_package(Boost REQUIRED)

find_package( OpenCV 3 REQUIRED )

################################################
## Declare ROS messages, services and actions ##
################################################



catkin_package(
INCLUDE_DIRS include
)

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})



###########
## Build ##
###########


## Declare a C++ executable
 add_executable(${PROJECT_NAME} src/klt_ros_driver.cpp)

add_dependencies(${PROJECT_NAME} sensor_msgs_gencpp)
target_link_libraries(${PROJECT_NAME}
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS} )