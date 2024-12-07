cmake_minimum_required(VERSION 3.0.2)
project(detect)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  sensor_msgs
  std_msgs
  resource_retriever
)

find_package(OpenCV REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES detect
#  CATKIN_DEPENDS cv_bridge roscpp rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(detect_template src/detect_template.cpp)
target_link_libraries(detect_template  ${catkin_LIBRARIES} ${OpenCV_LIBS})

install(TARGETS detect_template DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY resource
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
