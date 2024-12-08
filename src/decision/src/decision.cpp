#include "decision/decision.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>

using namespace decision;

cv::Point2d tf2_to_cv(const tf2::Transform& tf, const nav_msgs::OccupancyGrid& map)
{
    cv::Point2d map_origin(map.info.origin.position.x, map.info.origin.position.y);
    return cv::Point2d((tf.getOrigin().x() - map_origin.x) / map.info.resolution,
        (tf.getOrigin().y() - map_origin.y) / map.info.resolution);
}

void cv_set_tf2_translation(tf2::Transform& tf, const cv::Point2d& point, const nav_msgs::OccupancyGrid& map)
{
    tf.setOrigin(tf2::Vector3(point.x * map.info.resolution + map.info.origin.position.x,
        point.y * map.info.resolution + map.info.origin.position.y, 0));
}

cv::Point2d tf2_get_x_axis(const tf2::Transform& tf)
{
    cv::Point2d x(tf.getBasis()[0][0], tf.getBasis()[1][0]);
    return x / cv::norm(x);
}

void yaw_set_tf2(tf2::Transform& tf, double yaw)
{
    tf.setRotation(tf2::Quaternion(tf2::Vector3(0, 0, 1), yaw));
}

void Decision::update(const nav_msgs::OccupancyGrid& map, const geometry_msgs::TransformStamped& bot_transform)
{
    cv::Point2d map_origin(map.info.origin.position.x, map.info.origin.position.y);
    cv::Mat map_mat = cv::Mat(map.info.height, map.info.width, CV_8UC1);
    std::copy(map.data.begin(), map.data.end(), map_mat.data);

    // direction
    tf2::Transform bot_tf, last_goal_tf;
    tf2::fromMsg(bot_transform.transform, bot_tf);
    tf2::fromMsg(last_goal_, last_goal_tf);

    // std::cout << transform_tf.getOrigin().x() << " " << transform_tf.getOrigin().y() << std::endl;
    cv::Point2d bot_location = tf2_to_cv(bot_tf, map);
    cv::Point2d bot_x_axis = tf2_get_x_axis(bot_tf);
    double yaw = atan2(bot_x_axis.y, bot_x_axis.x);

    cv::Point2d goal_location = tf2_to_cv(last_goal_tf, map);
    cv::Point2d goal_x_axis = tf2_get_x_axis(last_goal_tf);

    cv::Mat border_map = cv::Mat::zeros(map_mat.size(), CV_8UC1);
    border_map.setTo(cv::Scalar(255), map_mat == 100);

state_reselect:
    switch (state_) {
    case State::S0_INIT:
        std::cout << "now: S0_INIT" << std::endl;
    case State::S1_STRAIGHT: {
        std::cout << "now: S1_STRAIGHT" << std::endl;
        if (state_ == State::S0_INIT) {
            state_ = State::S1_STRAIGHT;
        } else if ((bot_tf.getOrigin() - last_goal_tf.getOrigin()).length() < config_.goal_distance) {
            state_ = State::S2_TURN_DECISION;
            goto state_reselect;
        }
        cv::Vec2d edge = find_straight_edge(border_map, yaw);
        cv::Point2f goal_cv = find_border_infront(border_map, bot_location, bot_x_axis, edge);
        tf2::Transform goal_tf;
        cv_set_tf2_translation(goal_tf, goal_cv, map);
        yaw_set_tf2(goal_tf, yaw);
        tf2::toMsg(goal_tf, last_goal_);
        break;
    }
    case State::S2_TURN_DECISION: {
        std::cout << "now: S2_TURN_DECISION" << std::endl;
        if (turn_direction_ == TurnDirection::UNKNOWN) {
            std::cout << "turn direction unknown" << std::endl;
            break;
        }

        cv::Point2d straight = goal_x_axis;
        cv::Point2d turn;
        double new_yaw;
        switch (turn_direction_) {
        case TurnDirection::LEFT:
            turn = cv::Point2d(-goal_x_axis.y, goal_x_axis.x);
            new_yaw = yaw + CV_PI / 2;
            std::cout << "turn left" << std::endl;
            break;
        case TurnDirection::RIGHT:
            turn = cv::Point2d(goal_x_axis.y, -goal_x_axis.x);
            new_yaw = yaw - CV_PI / 2;
            std::cout << "turn right" << std::endl;
            break;
        default:
            std::cout << "turn direction error" << std::endl;
            break;
        }
        cv::Point2d new_goal_location = goal_location + straight * config_.turn_straight_distance + turn * config_.turn_turn_distance;
        tf2::Transform new_goal_tf;
        cv_set_tf2_translation(new_goal_tf, new_goal_location, map);
        yaw_set_tf2(new_goal_tf, new_yaw);
        tf2::toMsg(new_goal_tf, last_goal_);
        state_ = State::S3_STOP;
        break;
    }
    case State::S3_STOP: {
        std::cout << "now: S3_STOP" << std::endl;
        break;
    }
    }

    cv::Point2d new_goal_location_cv = tf2_to_cv(last_goal_tf, map);
    cv::Point2d new_goal_x_axis = tf2_get_x_axis(last_goal_tf);

    cv::Mat map_show;
    cv::cvtColor(map_mat, map_show, cv::COLOR_GRAY2BGR);

    cv::circle(map_show, bot_location, 5, cv::Scalar(0, 0, 255), -1);
    cv::arrowedLine(map_show, bot_location, bot_location + bot_x_axis * 20, cv::Scalar(0, 0, 255), 2);
    cv::circle(map_show, new_goal_location_cv, 5, cv::Scalar(0, 255, 0), -1);
    cv::arrowedLine(map_show, new_goal_location_cv, new_goal_location_cv + new_goal_x_axis * 20, cv::Scalar(0, 255, 0), 2);

    cv::flip(map_show, map_show, 0);
    cv::imshow("map", map_show);
}

void Decision::set_config(const DecisionConfig& config)
{
    config_ = config;
}

bool Decision::get_goal(geometry_msgs::Pose& goal)
{
    if (state_ == State::S0_INIT)
        return false;
    goal = last_goal_;
    return true;
}

void Decision::update_turn_direction(TurnDirection direction)
{
    turn_direction_ = direction;
}
