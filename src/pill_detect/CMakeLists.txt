cmake_minimum_required(VERSION 3.0.2)
project(pill_detect)

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
  ${PROJECT_SOURCE_DIR}/include
)

include_directories(${OpenCV_INCLUDE_DIRS})

# set(CMAKE_BUILD_TYPE Debug)
add_executable(pill_detect
  src/main.cpp
  src/match.cpp
  src/pill_detect.cpp
)
# target_compile_options(detect_template PRIVATE -O0 -fno-omit-frame-pointer)
target_link_libraries(pill_detect  ${catkin_LIBRARIES} ${OpenCV_LIBS})

install(TARGETS pill_detect DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY resource
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
