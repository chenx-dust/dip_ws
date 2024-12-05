#include "ground_detector/perspective_trans.hpp"

#include <resource_retriever/retriever.h>
#include <yaml-cpp/yaml.h>
#include <string>

using namespace ground_detector;

void PerspectiveTrans::showPerspectiveSelect(const cv::Mat& cv_image)
{
    cv::Mat image_show;
    if (paused_) {
        if (!already_paused_) {
            paused_image_ = cv_image.clone();
            already_paused_ = true;
        }
        image_show = paused_image_.clone();
    } else
        image_show = cv_image.clone();
    for (int i = 0; i < 4; i++) {
        if (i == selecting_index_) {
            cv::circle(image_show, selected_points_[i], 5, cv::Scalar(0, 0, 255), -1);
        } else {
            cv::circle(image_show, selected_points_[i], 5, cv::Scalar(0, 255, 0), -1);
        }
    }
    cv::imshow("Perspective Select", image_show);
}

void PerspectiveTrans::setNowPoint(const cv::Point2f& point)
{
    selected_points_[selecting_index_] = point;
    if (selecting_index_ == selected_points_.size() - 1) {
        perspective_matrix_ = cv::getPerspectiveTransform(selected_points_, corresponding_points_);
        std::cout << "Perspective Matrix: " << perspective_matrix_ << std::endl;
    }
    selecting_index_ = (selecting_index_ + 1) % config.target_points.size();
}


cv::Mat PerspectiveTrans::getPerspectiveMatrix() const
{
    return perspective_matrix_;
}

cv::Mat PerspectiveTrans::transform(const cv::Mat& cv_image, bool show_result)
{
    cv::Mat result;
    cv::warpPerspective(cv_image, result, perspective_matrix_, config.image_size);
    if (show_result) {
        cv::Mat result_show;
        if (result.channels() == 1)
            cv::cvtColor(result, result_show, cv::COLOR_GRAY2BGR);
        else
            result_show = result.clone();
        for (const auto& pt : corresponding_points_) {
            cv::circle(result_show, pt, 5, cv::Scalar(0, 255, 255), -1);
        }
        cv::imshow("Transformed", result_show);
    }
    return result;
}

cv::Point2f PerspectiveTrans::transform(const cv::Point2f point)
{
    cv::Point2f result;
    cv::Mat point_mat(1, 1, CV_64FC2);
    point_mat.at<cv::Vec2d>(0, 0) = cv::Vec2d(point.x, point.y);
    cv::perspectiveTransform(point_mat, point_mat, perspective_matrix_);
    result.x = point_mat.at<cv::Vec2d>(0, 0)[0];
    result.y = point_mat.at<cv::Vec2d>(0, 0)[1];
    return result;
}

void PerspectiveTrans::pauseOrResumeSelect()
{
    paused_ = !paused_;
    already_paused_ = false;
}

void PerspectiveTrans::loadConfig(const std::string& config_path)
{
    resource_retriever::Retriever retriever;
    auto config_str = retriever.get(config_path);
    std::string config_str_data(config_str.data.get(), config_str.data.get() + config_str.size);
    YAML::Node yaml = YAML::Load(config_str_data);

    config.image_size = cv::Size(yaml["image_size"][0].as<int>(), yaml["image_size"][1].as<int>());
    config.origin_point = cv::Point2f(yaml["origin_point"][0].as<float>(), yaml["origin_point"][1].as<float>());
    for (const auto& pt : yaml["target_points"]) {
        config.target_points.emplace_back(pt[0].as<float>(), pt[1].as<float>());
        corresponding_points_.emplace_back(pt[0].as<float>() + config.origin_point.x, config.image_size.height - pt[1].as<float>() - config.origin_point.y);
        selected_points_.emplace_back();
    }

    config.transform = cv::Mat::zeros(3, 3, CV_64FC1);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            config.transform.at<double>(i, j) = yaml["transform"][i][j].as<double>();
    perspective_matrix_ = config.transform.clone();
}

const PerspectiveTrans::PerspectiveConfig& PerspectiveTrans::getConfig() const
{
    return config;
}
