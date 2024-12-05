#pragma once

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

namespace ground_detector {
class RedDetection {
public:
    RedDetection();

    cv::Mat getRedMask(const cv::Mat& cv_image);

private:
    cv::Scalar red_lower_, red_upper_;

    // 定义滑动条的初始值
    int lower_h_ = 0, lower_s_ = 0, lower_v_ = 100; // 低阈值
    int upper_h_ = 100, upper_s_ = 80, upper_v_ = 255; // 高阈值
};
}
