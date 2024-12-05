#include "map_registration/global_map.hpp"
#include "map_registration/registration.hpp"
#include <tf2_eigen/tf2_eigen.h>
#include <opencv2/reg/mappergradproj.hpp>

using namespace map_registration;

GlobalMap::GlobalMap(const MapConfig& config)
    : config_(config)
{
}

void GlobalMap::updateMap(const nav_msgs::OccupancyGrid& local_map, const geometry_msgs::TransformStamped& transform)
{
    cv::Mat local_map_mat = cv::Mat(local_map.info.height, local_map.info.width, CV_8UC1);
    std::copy(local_map.data.begin(), local_map.data.end(), local_map_mat.data);
    // cv::imshow("local_map", local_map_mat);

    cv::Mat road = local_map_mat == 0;
    cv::imshow("road", road);
    cv::Mat border = local_map_mat == 100;
    cv::imshow("border", border);

    // Create a transformation matrix from the tf2 transform
    tf2::Transform tf2_transform;
    tf2::fromMsg(transform.transform, tf2_transform);

    Eigen::Vector2d local_origin(local_map.info.origin.position.x, local_map.info.origin.position.y);
    Eigen::Matrix2d rotation_matrix;
    rotation_matrix << tf2_transform.getBasis()[0][0], tf2_transform.getBasis()[0][1],
                       tf2_transform.getBasis()[1][0], tf2_transform.getBasis()[1][1];
    Eigen::Vector2d transformed_origin = rotation_matrix * local_origin;
    // Convert the tf2 transform to an OpenCV affine transformation matrix
    cv::Mat transform_mat = (cv::Mat_<double>(2, 3) << 
        tf2_transform.getBasis()[0][0], tf2_transform.getBasis()[0][1],
            (tf2_transform.getOrigin().x() + transformed_origin[0]) / config_.resolution + config_.origin.x,
        tf2_transform.getBasis()[1][0], tf2_transform.getBasis()[1][1],
            (tf2_transform.getOrigin().y() + transformed_origin[1]) / config_.resolution + config_.origin.y);

    cv::Mat last_registration_ = (cv::Mat_<double>(2, 3) << 
        last_transform_.getBasis()[0][0], last_transform_.getBasis()[0][1],
            last_transform_.getOrigin().x() / config_.resolution,
        last_transform_.getBasis()[1][0], last_transform_.getBasis()[1][1],
            last_transform_.getOrigin().y() / config_.resolution);

    // std::cout << "last mat: " << last_registration_ << std::endl;

    // Apply the affine transformation to the local map
    cv::Mat transformed_road;
    cv::warpAffine(road, transformed_road, transform_mat, config_.size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::warpAffine(transformed_road, transformed_road, last_registration_, config_.size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::threshold(transformed_road, transformed_road, 127, 255, cv::THRESH_BINARY);

    cv::Mat transformed_border;
    cv::warpAffine(border, transformed_border, transform_mat, config_.size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::warpAffine(transformed_border, transformed_border, last_registration_, config_.size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::threshold(transformed_border, transformed_border, 127, 255, cv::THRESH_BINARY);

    // Update the global map with the transformed local map
    if (border_.empty()) {
        border_ = transformed_border;
        road_ = transformed_road;
    } else {
        // cv::Mat old_border_gauss, new_border_gauss;
        // cv::GaussianBlur(border_, old_border_gauss, cv::Size(7, 7), 1);
        // cv::GaussianBlur(transformed_border, new_border_gauss, cv::Size(7, 7), 1);
        // cv::imshow("old_border_gauss", old_border_gauss);
        // cv::imshow("new_border_gauss", new_border_gauss);
        // cv::reg::MapperGradProj mapper;
        // auto map = mapper.calculate(new_border_gauss, old_border_gauss);

        cv::Mat warped_border, warped_road;
        cv::Mat result = registration(border_, transformed_border);

        // std::cout << "new mat: " << result << std::endl;
        // map->warp(transformed_border, warped_border);
        // map->warp(transformed_road, warped_road);

        // icp icp_;
        // cv::Mat result = icp_.registration(border_, transformed_border, 100, 0.01);

        // cv::warpAffine(now_registration, warped_registration, result, config_.size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::warpAffine(transformed_border, warped_border, result, config_.size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::threshold(warped_border, warped_border, 127, 255, cv::THRESH_BINARY);
        cv::warpAffine(transformed_road, warped_road, result, config_.size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::threshold(warped_road, warped_road, 127, 255, cv::THRESH_BINARY);

        // cv::Mat border_show;
        // cv::cvtColor(registration_, border_show, cv::COLOR_GRAY2BGR);
        // border_show.setTo(cv::Scalar(0, 255, 255), warped_registration);
        // cv::flip(border_show, border_show, 0);
        // cv::imshow("border_show", border_show);
        cv::Mat warped_registration = cv::Mat::zeros(config_.size, CV_8UC1);
        warped_registration.setTo(cv::Scalar(255), warped_border);
        warped_registration.setTo(cv::Scalar(50), warped_road);
        // warped_registration.copyTo(registration_, warped_registration != 0);

        cv::Mat registration_show;

        cv::Mat now_registration = cv::Mat::zeros(config_.size, CV_8UC1);
        now_registration.setTo(cv::Scalar(50), road_);
        now_registration.setTo(cv::Scalar(255), border_);
        cv::cvtColor(now_registration, registration_show, cv::COLOR_GRAY2BGR);
        registration_show.setTo(cv::Scalar(0, 255, 255), warped_border);
        registration_show.setTo(cv::Scalar(0, 0, 255), transformed_border);
        cv::flip(registration_show, registration_show, 0);
        cv::imshow("registration_show", registration_show);

        cv::Mat eroded_road;
        cv::Mat morph_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::erode(warped_road, eroded_road, morph_kernel);

        border_.setTo(cv::Scalar(0), eroded_road);
        border_ += warped_border;
        road_ += warped_road;


        last_registration_ = result;

        tf2::Matrix3x3 rotation_matrix(result.at<double>(0, 0), result.at<double>(0, 1), 0,
            result.at<double>(1, 0), result.at<double>(0, 0), 0,
            0, 0, 1);
        tf2::Vector3 translation(result.at<double>(0, 2), result.at<double>(1, 2), 0);
        last_transform_ = tf2::Transform(rotation_matrix, translation * config_.resolution) * last_transform_;
    }

    cv::Mat border_draw;
    cv::cvtColor(border_, border_draw, cv::COLOR_GRAY2BGR);
    cv::circle(border_draw, cv::Point2f(tf2_transform.getOrigin().x() / config_.resolution, tf2_transform.getOrigin().y() / config_.resolution) + config_.origin,
        10, cv::Scalar(0, 0, 255), -1);
    // flip to make origin at the bottom left
    cv::flip(border_draw, border_draw, 0);
    cv::imshow("border_draw", border_draw);

    cv::Mat road_draw = road_.clone();
    // cv::cvtColor(road_, road_draw, cv::COLOR_GRAY2BGR);
    cv::flip(road_draw, road_draw, 0);
    cv::imshow("road_draw", road_draw);
}

tf2::Transform GlobalMap::getTransform() const
{
    return last_transform_;
}

const cv::Mat& GlobalMap::getBorder() const
{
    return border_;
}

const cv::Mat& GlobalMap::getRoad() const
{
    return road_;
}

cv::Point2f GlobalMap::getOrigin() const
{
    return config_.origin;
}