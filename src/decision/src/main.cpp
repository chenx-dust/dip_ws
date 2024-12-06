#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <opencv2/opencv.hpp>

#include "decision/decision.hpp"

tf2_ros::Buffer tf_buffer;
decision::Decision decision_;

void map_callback(const nav_msgs::OccupancyGridConstPtr &msg) {
    try {
        // Get the transform at the time of the map message
        geometry_msgs::TransformStamped transform_stamped =
            tf_buffer.lookupTransform("map", "base_footprint", msg->header.stamp, ros::Duration(0.1));

        // Update global map with the local map and its transform
        decision_.update(*msg, transform_stamped);

        cv::waitKey(1);

    } catch (tf2::TransformException& ex) {
        ROS_WARN("Failed to get transform: %s", ex.what());
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "decision");
    ros::NodeHandle nh;

    tf2_ros::TransformListener tf_listener(tf_buffer);
    decision_.set_config({
        .angle_threshold = CV_PI / 6,
        .hough_threshold = 80,
        .rho_threshold = 40,
        .front_length = 80,
    });

    ros::Subscriber map_sub = nh.subscribe("/global_map", 10, map_callback);
    ros::spin();
    return 0;
}