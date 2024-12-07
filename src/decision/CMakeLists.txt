cmake_minimum_required(VERSION 3.0.2)
project(decision)

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
  tf2
  tf2_geometry_msgs
  tf2_ros
  tf2_eigen
)

find_package(OpenCV REQUIRED)

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

add_executable(decision
  src/main.cpp
  src/decision.cpp
  src/process.cpp
)
target_link_libraries(decision ${catkin_LIBRARIES} ${OpenCV_LIBS})

install(TARGETS decision
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
