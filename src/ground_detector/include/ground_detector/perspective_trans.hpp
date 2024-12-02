#pragma once

#include <opencv2/opencv.hpp>


class PerspectiveTrans {
public:
    void setNowPoint(const cv::Point2f& point);

    void showPerspectiveSelect(const cv::Mat& cv_image);

    cv::Mat getPerspectiveMatrix() const;

    cv::Mat transform(const cv::Mat& cv_image, bool show_result = false);

    void pauseOrResumeSelect();

    void loadConfig(const std::string& config_path);
    
    struct PerspectiveConfig {
        cv::Size image_size;
        cv::Point2f origin_point;
        cv::Mat transform;
        std::vector<cv::Point2f> target_points;
    };

    const PerspectiveConfig& getConfig() const;

private:
    int selecting_index_ = 0;
    std::vector<cv::Point2f> selected_points_;
    std::vector<cv::Point2f> corresponding_points_;

    bool paused_ = false;
    bool already_paused_ = false;
    cv::Mat paused_image_;
    cv::Mat perspective_matrix_;

    PerspectiveConfig config;
};

