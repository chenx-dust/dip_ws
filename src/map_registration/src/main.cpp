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
ros::Publisher map_pub;

void publishTransform(const tf2::Transform& transform, const ros::Time& stamp) {
    geometry_msgs::TransformStamped transform_msg;
    transform_msg.transform = tf2::toMsg(transform);
    transform_msg.header.stamp = stamp;
    transform_msg.header.frame_id = "map";
    transform_msg.child_frame_id = "odom_combined";
    transform_broadcaster_ptr->sendTransform(transform_msg);
}

void publishMap(ros::Time stamp, const cv::Mat& road_mask, const cv::Mat& border_mask, cv::Point2f origin_point)
{
    nav_msgs::OccupancyGrid map_msg;
    map_msg.header.stamp = stamp;
    map_msg.header.frame_id = "map";
    map_msg.info.resolution = 0.01;
    map_msg.info.width = road_mask.cols;
    map_msg.info.height = road_mask.rows;
    map_msg.info.origin.position.x = -origin_point.x / 100.;
    map_msg.info.origin.position.y = -origin_point.y / 100.;
    map_msg.info.origin.position.z = 0;
    map_msg.info.origin.orientation.x = 0;
    map_msg.info.origin.orientation.y = 0;
    map_msg.info.origin.orientation.z = 0;
    map_msg.info.origin.orientation.w = 1;
    map_msg.data.resize(road_mask.cols * road_mask.rows);

    for (int i = 0; i < road_mask.rows; i++) {
        for (int j = 0; j < road_mask.cols; j++) {
            size_t index = i * road_mask.cols + j;
            if (border_mask.at<uchar>(i, j) == 255) {
                map_msg.data[index] = 100;
            } else {
                map_msg.data[index] = 255 - road_mask.at<uchar>(i, j);
            }
        }
    }
    map_pub.publish(map_msg);
}

// Callback function for OccupancyGrid messages
void localMapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
    try {
        // Get the transform at the time of the map message
        geometry_msgs::TransformStamped transform_stamped = 
            tf_buffer.lookupTransform("odom_combined", msg->header.frame_id, msg->header.stamp, ros::Duration(0.1));

        // Update global map with the local map and its transform
        if (global_map_ptr) {
            global_map_ptr->updateMap(*msg, transform_stamped);
            publishTransform(global_map_ptr->getTransform(), msg->header.stamp);
            publishMap(msg->header.stamp, global_map_ptr->getRoad(), global_map_ptr->getBorder(), global_map_ptr->getOrigin());
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
    map_pub = nh.advertise<nav_msgs::OccupancyGrid>("/global_map", 1);

    // publishTransform(tf2::Transform::getIdentity(), ros::Time::now());

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
