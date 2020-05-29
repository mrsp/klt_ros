cmake_minimum_required(VERSION 2.8.3)
project(klt_ros)


add_compile_options(-std=c++14 -fopenmp)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  geometry_msgs
  image_transport
  cv_bridge
  message_filters
)
SET(teaser_src
    ${CMAKE_CURRENT_LIST_DIR}/src/teaser/graph.cc
#     ${CMAKE_CURRENT_LIST_DIR}/src/teaser/matcher.cc
    ${CMAKE_CURRENT_LIST_DIR}/src/teaser/registration.cc
)

SET(pmc_src
    ${CMAKE_CURRENT_LIST_DIR}/src/pmc/pmc_clique_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/pmc/pmc_cores.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/pmc/pmc_graph.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/pmc/pmc_heu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/pmc/pmc_lib.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/pmc/pmc_maxclique.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/pmc/pmc_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/pmc/pmcx_maxclique_basic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/pmc/pmcx_maxclique.cpp
)



## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 REQUIRED)
# find_package(teaserpp REQUIRED)


find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")


################################################
## Declare ROS messages, services and actions ##
################################################



 catkin_package(
 INCLUDE_DIRS include 
 LIBRARIES ${PROJECT_NAME}
 CATKIN_DEPENDS roscpp message_filters
 )

include_directories(include ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}  
    ${CMAKE_CURRENT_LIST_DIR}/src/teaser
    ${CMAKE_CURRENT_LIST_DIR}/src/pmc
    ${CMAKE_CURRENT_LIST_DIR}/src/
    ${EIGEN3_INCLUDE_DIR}
)



###########
## Build ##
###########


## Declare a C++ executable
add_executable(${PROJECT_NAME} src/klt_ros_driver.cpp src/klt_ros.cpp ${teaser_src} ${pmc_src})


target_link_libraries(${PROJECT_NAME}
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${OpenCV_LIBS})
add_dependencies(${PROJECT_NAME} sensor_msgs_gencpp)