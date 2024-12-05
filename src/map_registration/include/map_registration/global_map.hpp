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
    };

    GlobalMap(const MapConfig& config);

    tf2::Transform updateMap(const nav_msgs::OccupancyGrid& local_map, const geometry_msgs::TransformStamped& transform);
private:
    MapConfig config_;
    cv::Mat registration_;
    cv::Mat border_;
    cv::Mat road_;
    tf2::Transform transform_ = tf2::Transform::getIdentity();
    int count_ = 0;
};
}
