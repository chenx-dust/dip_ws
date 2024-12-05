#include "ground_detector/red_extract.hpp"
#include "ground_detector/perspective_trans.hpp"
#include "ground_detector/road_extract.hpp"
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <camera_info_manager/camera_info_manager.h>
#include <image_geometry/pinhole_camera_model.h>
#include <nav_msgs/OccupancyGrid.h>

using namespace ground_detector;

RedDetection red_detection;
PerspectiveTrans perspective_trans;
std::shared_ptr<camera_info_manager::CameraInfoManager> camera_info;
image_geometry::PinholeCameraModel model;

ros::Publisher map_pub;

void mouseCallback(int event, int x, int y, int flags, void* userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        perspective_trans.setNowPoint(cv::Point2f(x, y));
    }
}

void publishMap(ros::Time stamp, const cv::Mat& road_mask, const cv::Mat& border_mask, cv::Point2f origin_point)
{
    nav_msgs::OccupancyGrid msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = "base_footprint";
    msg.info.resolution = 0.01;
    msg.info.width = road_mask.cols;
    msg.info.height = road_mask.rows;
    msg.info.origin.position.x = -origin_point.y / 100.;
    msg.info.origin.position.y = -origin_point.x / 100.;
    msg.info.origin.position.z = 0;
    msg.info.origin.orientation.x = 0;
    msg.info.origin.orientation.y = 0;
    msg.info.origin.orientation.z = 0;
    msg.info.origin.orientation.w = 1;
    msg.data.resize(road_mask.cols * road_mask.rows);
    for (int i = 0; i < road_mask.rows; i++) {
        for (int j = 0; j < road_mask.cols; j++) {
            size_t index = msg.data.size() - (j * road_mask.cols + i);
            if (border_mask.at<uchar>(i, j) == 255) {
                msg.data[index] = 100;
            } else {
                msg.data[index] = 255 - road_mask.at<uchar>(i, j);
            }
        }
    }
    map_pub.publish(msg);
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv::imshow("Raw", cv_ptr->image);

    cv::Mat rectified_image;
    model.rectifyImage(cv_ptr->image, rectified_image);
    cv::imshow("Rectified", rectified_image);

    perspective_trans.showPerspectiveSelect(rectified_image);
    perspective_trans.transform(rectified_image, true);

    cv::Mat red_mask = red_detection.getRedMask(rectified_image);

    cv::Mat transformed_red_mask = perspective_trans.transform(255 - red_mask);
    cv::Mat road_mask = roadExtract(255 - transformed_red_mask);
    cv::Mat border_mask = borderExtract(road_mask, perspective_trans.getPerspectiveMatrix(), rectified_image.size());

    int key = cv::waitKey(1);
    if (key == 'p') {
        perspective_trans.pauseOrResumeSelect();
    }
    publishMap(ros::Time::now(), road_mask, border_mask, perspective_trans.getConfig().origin_point);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ground_detector");
    ros::NodeHandle nh;

    camera_info = std::make_shared<camera_info_manager::CameraInfoManager>(nh);
    camera_info->loadCameraInfo("package://ground_detector/config/camera_info.yaml");

    model.fromCameraInfo(camera_info->getCameraInfo());

    perspective_trans.loadConfig("package://ground_detector/config/perspective.yaml");

    map_pub = nh.advertise<nav_msgs::OccupancyGrid>("/local_map", 1);
    ros::Subscriber sub = nh.subscribe("/camera/color/image_raw", 1, imageCallback);
    cv::namedWindow("Perspective Select");
    cv::setMouseCallback("Perspective Select", mouseCallback, NULL);
    ros::spin();

    return 0;
}