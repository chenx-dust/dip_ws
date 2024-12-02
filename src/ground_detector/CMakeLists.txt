cmake_minimum_required(VERSION 3.0.2)
project(ground_detector)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  sensor_msgs
  std_msgs
  # image_transport
  camera_info_manager
  image_geometry
  resource_retriever
  nav_msgs
)

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

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
  ${PROJECT_SOURCE_DIR}/include
)

add_executable(ground_detector
  src/main.cpp
  src/red_extract.cpp
  src/perspective_trans.cpp
  src/road_extract.cpp
)
target_link_libraries(ground_detector ${catkin_LIBRARIES} ${OpenCV_LIBS} ${YAML_CPP_LIBRARIES})

install(TARGETS ground_detector
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

