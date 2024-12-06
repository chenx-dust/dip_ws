#include "decision/decision.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>

using namespace decision;

void Decision::update(const nav_msgs::OccupancyGrid& map, const geometry_msgs::TransformStamped& transform)
{
    cv::Point2d map_origin(map.info.origin.position.x, map.info.origin.position.y);
    cv::Mat map_mat = cv::Mat(map.info.height, map.info.width, CV_8UC1);
    std::copy(map.data.begin(), map.data.end(), map_mat.data);

    // direction
    tf2::Transform transform_tf;
    tf2::fromMsg(transform.transform, transform_tf);

    // std::cout << transform_tf.getOrigin().x() << " " << transform_tf.getOrigin().y() << std::endl;
    cv::Point2d bot_location((transform_tf.getOrigin().x() - map_origin.x) / map.info.resolution,
                             (transform_tf.getOrigin().y() - map_origin.y) / map.info.resolution);
    cv::Point2d x_axis(transform_tf.getBasis()[0][0], transform_tf.getBasis()[1][0]);

    cv::Mat border_map = cv::Mat::zeros(map_mat.size(), CV_8UC1);
    border_map.setTo(cv::Scalar(255), map_mat == 100);
    double yaw = atan2(x_axis.y, x_axis.x);
    cv::Vec2d edge = find_edge(border_map, yaw);
    cv::Point2f border = find_border_infront(border_map, bot_location, x_axis, edge);

    cv::Mat map_show;
    cv::cvtColor(map_mat, map_show, cv::COLOR_GRAY2BGR);

    cv::circle(map_show, bot_location, 5, cv::Scalar(0, 0, 255), -1);
    cv::arrowedLine(map_show, bot_location, bot_location + x_axis * 20, cv::Scalar(0, 0, 255), 2);

    cv::flip(map_show, map_show, 0);
    cv::imshow("map", map_show);
}

void Decision::set_config(const DecisionConfig& config)
{
    config_ = config;
}
