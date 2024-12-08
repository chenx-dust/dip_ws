#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

namespace pill_detect {

constexpr int MAX_BLUE_COUNT = 10;
constexpr int MAX_GREEN_COUNT = 10;

class PillDetect {
public:
    struct DetectConfig {
        // 初始参数
        int threshold_value = 175;
        int min_area = 700;
        double min_aspect_ratio = 0.50;
        double max_aspect_ratio = 1.0;
        double max_angle_horizon = 10;
        double max_angle_diff = 10.0; // 允许的最大内角度偏差（接近直角）
        double single_scale = 1.1;
        double score_thres_blue = 0.8;
        double score_thres_green = 0.8;
        double iou_thres_blue = 0.1;
        double iou_thres_green = 0.1;
        int pill_min_area = 10; // 药片最小面积阈值
        int diff_threshold = 30;
        int erode_kernel_size = 7;
        int erode_iteration = 3;
        int open_kernel_size = 7;
        int open_iteration = 3;
        std::vector<int> scale_num_blue = {-5, 3};
        std::vector<int> scale_num_green = {-5, 3};
    };

    enum class PillType {
        UNKNOWN = 0,
        BLUE = 1,
        GREEN = 2
    };

    std::pair<PillType, int> detect_pill_type(const cv::Mat& img);
    void set_template(const cv::Mat& templ_blue, const cv::Mat& templ_green);
    void set_config(const DetectConfig& config);

private:
    DetectConfig config_;
    bool color_decision = false;
    bool is_blue = false;

    cv::Mat templ_blue_;
    cv::Mat templ_green_;

    size_t total_blue_count_ = 0;
    size_t total_green_count_ = 0;

};
} // namespace pill_detect
