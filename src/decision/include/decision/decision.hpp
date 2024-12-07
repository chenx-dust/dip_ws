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

        double goal_distance;
        double turn_straight_distance;
        double turn_turn_distance;
    };

    enum class State {
        S0_INIT,
        S1_STRAIGHT,
        S2_TURN_DECISION,
        S3_STOP,
    };

    enum class TurnDirection {
        UNKNOWN = 0,
        LEFT = 1,
        RIGHT = 2,
    };

    void set_config(const DecisionConfig& config);
    void update(const nav_msgs::OccupancyGrid& map, const geometry_msgs::TransformStamped& transform);
    void update_turn_direction(TurnDirection direction);
    bool get_goal(geometry_msgs::Pose& goal);

private:
    DecisionConfig config_;
    State state_ = State::S0_INIT;
    TurnDirection turn_direction_ = TurnDirection::UNKNOWN;
    geometry_msgs::Pose last_goal_;
    cv::Vec2d find_straight_edge(const cv::Mat& map, double yaw);
    cv::Point2f find_border_infront(const cv::Mat& image, cv::Point bot_location, cv::Point bot_direction, cv::Vec2d line);
};
}
