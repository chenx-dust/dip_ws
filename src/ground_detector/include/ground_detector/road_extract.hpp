#pragma once

#include <opencv2/opencv.hpp>

namespace ground_detector {
cv::Mat roadExtract(const cv::Mat& cv_image);

cv::Mat borderExtract(const cv::Mat& cv_image, const cv::Mat& perspective_matrix, cv::Size origin_size);
}
