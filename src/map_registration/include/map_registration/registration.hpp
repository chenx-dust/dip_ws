#pragma once

#include <opencv2/core.hpp>

namespace map_registration {
cv::Mat registration(const cv::Mat& src, const cv::Mat& dst);
}

