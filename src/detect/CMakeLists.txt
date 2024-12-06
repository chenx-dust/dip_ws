cmake_minimum_required(VERSION 3.0.2)
project(detect)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

find_package(OpenCV 3.4.5 REQUIRED)

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

add_executable(detect_hsv src/detect_hsv.cpp)
target_link_libraries(detect_hsv  ${catkin_LIBRARIES} ${OpenCV_LIBS})


add_executable(detect_template src/detect_template.cpp)
target_link_libraries(detect_template  ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(detect_template_kcf src/detect_template_kcf.cpp)
target_link_libraries(detect_template_kcf  ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(detect_template_kcf_mix src/detect_template_kcf_mix.cpp)
target_link_libraries(detect_template_kcf_mix  ${catkin_LIBRARIES} ${OpenCV_LIBS})


