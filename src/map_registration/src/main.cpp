#include <ros/ros.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/OccupancyGrid.h>

#include "map_registration/global_map.hpp"

// Global variables
tf2_ros::Buffer tf_buffer;
std::shared_ptr<map_registration::GlobalMap> global_map_ptr;
std::shared_ptr<tf2_ros::StaticTransformBroadcaster> transform_broadcaster_ptr;

void publishTransform(const tf2::Transform& transform) {
    geometry_msgs::TransformStamped transform_msg;
    transform_msg.transform = tf2::toMsg(transform);
    transform_msg.header.stamp = ros::Time::now();
    transform_msg.header.frame_id = "map";
    transform_msg.child_frame_id = "odom_combined";
    transform_broadcaster_ptr->sendTransform(transform_msg);
}

// Callback function for OccupancyGrid messages
void localMapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
    try {
        // Get the transform at the time of the map message
        geometry_msgs::TransformStamped transform_stamped = 
            tf_buffer.lookupTransform("map", msg->header.frame_id, msg->header.stamp, ros::Duration(0.1));

        // Update global map with the local map and its transform
        if (global_map_ptr) {
            tf2::Transform transform = global_map_ptr->updateMap(*msg, transform_stamped);
            publishTransform(transform);
        }

        cv::waitKey(1);

    } catch (tf2::TransformException &ex) {
        ROS_WARN("Failed to get transform: %s", ex.what());
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "map_registration");
    ros::NodeHandle nh;

    // Initialize tf2 listener
    tf2_ros::TransformListener tf_listener(tf_buffer);
    transform_broadcaster_ptr = std::make_shared<tf2_ros::StaticTransformBroadcaster>();

    publishTransform(tf2::Transform::getIdentity());

    // Initialize global map (you'll need to set appropriate config values)
    map_registration::GlobalMap::MapConfig config
    {
        .size = cv::Size(nh.param<int>("width", 1000), nh.param<int>("height", 1000)),
        .resolution = nh.param<float>("resolution", 0.01),
        .origin = cv::Point2f(nh.param<float>("origin_x", 500), nh.param<float>("origin_y", 200))
    };
    // Set config values here...
    global_map_ptr = std::make_shared<map_registration::GlobalMap>(config);

    // Subscribe to the local map topic
    ros::Subscriber local_map_sub = nh.subscribe("/local_map", 1, localMapCallback);

    ros::spin();
    return 0;
}
