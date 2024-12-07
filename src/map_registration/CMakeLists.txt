cmake_minimum_required(VERSION 3.0.2)
project(map_registration)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  sensor_msgs
  std_msgs
  # image_transport
  camera_info_manager
  image_geometry
  nav_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  tf2_eigen
)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ITK REQUIRED COMPONENTS ITKVideoBridgeOpenCV)
find_package(Elastix REQUIRED)

include(${ITK_USE_FILE})
include(${ELASTIX_USE_FILE})

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ground_detector
#  CATKIN_DEPENDS cv_bridge opencv2 rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
)

add_executable(map_registration
  src/main.cpp
  src/global_map.cpp
  src/registration.cpp
)
target_link_libraries(map_registration
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${YAML_CPP_LIBRARIES}
  ${ITK_LIBRARIES}
  ${ELASTIX_LIBRARIES}
)

install(TARGETS map_registration
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

