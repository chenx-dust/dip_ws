#pragma once

#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/TransformStamped.h>
#include <opencv2/opencv.hpp>

namespace decision {
class Decision {
public:
    struct DecisionConfig {
        double angle_threshold;
        double hough_threshold;
        double rho_threshold;
        double front_length;
    };

    enum class State {
        S1_STRAIGHT,
        S2_TURNING,
        S3_STOP,
    };

    void set_config(const DecisionConfig& config);
    void update(const nav_msgs::OccupancyGrid& map, const geometry_msgs::TransformStamped& transform);

private:
    DecisionConfig config_;
    State state_;
    cv::Vec2d find_edge(const cv::Mat& map, double yaw);
    cv::Point2f find_border_infront(const cv::Mat& image, cv::Point bot_location, cv::Point bot_direction, cv::Vec2d line);
};
}
