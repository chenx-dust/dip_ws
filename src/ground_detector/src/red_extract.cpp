#include "ground_detector/red_extract.hpp"

RedDetection::RedDetection()
{
    // 设置红色的初始低阈值和高阈值（BGR格式）
    red_lower_ = cv::Scalar(0, 0, 100); // 红色的低阈值
    red_upper_ = cv::Scalar(220, 55, 255); // 红色的高阈值

    // 创建 OpenCV 窗口
    cv::namedWindow("Red Mask");
    cv::namedWindow("Red Detected");

    // 创建滑动条
    cv::createTrackbar("Lower Red H", "Red Mask", &lower_h_, 255);
    cv::createTrackbar("Lower Red S", "Red Mask", &lower_s_, 255);
    cv::createTrackbar("Lower Red V", "Red Mask", &lower_v_, 255);
    cv::createTrackbar("Upper Red H", "Red Mask", &upper_h_, 255);
    cv::createTrackbar("Upper Red S", "Red Mask", &upper_s_, 255);
    cv::createTrackbar("Upper Red V", "Red Mask", &upper_v_, 255);
}

cv::Mat RedDetection::getRedMask(const cv::Mat& cv_image)
{
    // 获取滑动条值并更新阈值
    red_lower_ = cv::Scalar(lower_h_, lower_s_, lower_v_);
    red_upper_ = cv::Scalar(upper_h_, upper_s_, upper_v_);

    cv::Mat gaussian_image;
    cv::GaussianBlur(cv_image, gaussian_image, cv::Size(7, 7), 0);

    // 提取红色部分
    cv::Mat mask;
    if (red_lower_(0) < red_upper_(0)) {
        cv::inRange(gaussian_image, red_lower_, red_upper_, mask);
    } else {
        cv::Scalar temp_lower = red_lower_, temp_upper = red_upper_;
        red_lower_(0) = 0;
        red_upper_(0) = 255;
        cv::Mat mask1, mask2;
        cv::inRange(gaussian_image, temp_lower, red_lower_, mask1);
        cv::inRange(gaussian_image, red_lower_, temp_upper, mask2);
        mask = mask1 | mask2;
    }

    // 使用形态学操作去除噪声
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    // 使用掩膜提取红色区域
    cv::Mat result;
    cv::bitwise_and(cv_image, cv_image, result, mask);

    // 显示图像
    cv::imshow("Red Mask", mask);
    cv::imshow("Red Detected", result);

    return mask;
}