#pragma once

#include <vector>
#include <utility>
#include <opencv2/opencv.hpp>

namespace pill_detect {

double angleBetweenVectors(const cv::Point& p1, const cv::Point& p2);
bool comparePairs(const std::pair<float, cv::Rect>& a, const std::pair<float, cv::Rect>& b);

std::vector<cv::Rect> nonMaxSuppression(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, float overlapThresh);
void multiScaleTemplateMatching(const cv::Mat& img, const cv::Mat& templ, std::vector<cv::Rect>& boxes, std::vector<float>& scores, double scaleFactor = 1.1, int minScale = 0, int maxScale = 3, double threshold = 0.8, float overlapThresh = 0.01, int maxMatches = 6);

} // namespace pill_detect
