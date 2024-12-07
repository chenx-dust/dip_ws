#pragma once

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace map_registration {
class GlobalMap {
public:
    struct MapConfig {
        cv::Size size;
        float resolution;
        cv::Point2f origin;
        float init_circle_radius;
        float max_reg_move;
    };

    GlobalMap(const MapConfig& config);

    void updateMap(const nav_msgs::OccupancyGrid& local_map, const geometry_msgs::TransformStamped& transform);

    tf2::Transform getTransform() const;

    const cv::Mat& getBorder() const;
    const cv::Mat& getRoad() const;
    cv::Point2f getOrigin() const;
private:
    MapConfig config_;
    cv::Mat border_;
    cv::Mat road_;

    // cv::Mat last_registration_ = cv::Mat::eye(2, 3, CV_32F);
    tf2::Transform last_transform_ = tf2::Transform::getIdentity();
    int count_ = 0;
};
}
