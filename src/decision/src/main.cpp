#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <opencv2/opencv.hpp>
#include <std_msgs/Int8.h>

#include "decision/decision.hpp"

tf2_ros::Buffer tf_buffer;
decision::Decision decision_;
ros::Publisher goal_pub;

void map_callback(const nav_msgs::OccupancyGridConstPtr &msg) {
    try {
        // Get the transform at the time of the map message
        geometry_msgs::TransformStamped transform_stamped =
            tf_buffer.lookupTransform("map", "base_footprint", msg->header.stamp, ros::Duration(0.1));

        // Update global map with the local map and its transform
        decision_.update(*msg, transform_stamped);

        cv::waitKey(1);

        geometry_msgs::PoseStamped goal_msg;
        if (decision_.get_goal(goal_msg.pose)) {
            goal_msg.header.stamp = msg->header.stamp;
            goal_msg.header.frame_id = "map";
            goal_pub.publish(goal_msg);
        }

    } catch (tf2::TransformException& ex) {
        ROS_WARN("Failed to get transform: %s", ex.what());
    }
}

void turn_direction_callback(const std_msgs::Int8ConstPtr &msg) {
    decision_.update_turn_direction(static_cast<decision::Decision::TurnDirection>(msg->data));
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "decision");
    ros::NodeHandle nh;

    tf2_ros::TransformListener tf_listener(tf_buffer);
    decision_.set_config({
        .angle_threshold = CV_PI / 6,
        .hough_threshold = 80,
        .rho_threshold = 40,
        .front_length = 100,
        .goal_distance = 0.75,
        .turn_straight_distance = 50,
        .turn_turn_distance = 150,
    });

    goal_pub = nh.advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal", 10);
    ros::Subscriber map_sub = nh.subscribe("/global_map", 10, map_callback);
    ros::Subscriber turn_direction_sub = nh.subscribe("/turn_direction", 10, turn_direction_callback);
    ros::spin();
    return 0;
}