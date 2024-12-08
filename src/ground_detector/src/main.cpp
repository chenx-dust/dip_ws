#include "ground_detector/red_extract.hpp"
#include "ground_detector/perspective_trans.hpp"
#include "ground_detector/road_extract.hpp"
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <camera_info_manager/camera_info_manager.h>
#include <image_geometry/pinhole_camera_model.h>
#include <nav_msgs/OccupancyGrid.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

using namespace ground_detector;

RedDetection red_detection;
PerspectiveTrans perspective_trans;
std::shared_ptr<camera_info_manager::CameraInfoManager> camera_info;
image_geometry::PinholeCameraModel model;

ros::Publisher map_pub;
ros::Publisher pc_pub;

void mouseCallback(int event, int x, int y, int flags, void* userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        perspective_trans.setNowPoint(cv::Point2f(x, y));
    }
}

void publishMap(ros::Time stamp, const cv::Mat& road_mask, const cv::Mat& border_mask, cv::Point2f origin_point)
{
    nav_msgs::OccupancyGrid map_msg;
    map_msg.header.stamp = stamp;
    map_msg.header.frame_id = "base_footprint";
    map_msg.info.resolution = 0.01;
    map_msg.info.width = road_mask.cols;
    map_msg.info.height = road_mask.rows;
    map_msg.info.origin.position.x = -origin_point.y / 100.;
    map_msg.info.origin.position.y = -origin_point.x / 100.;
    map_msg.info.origin.position.z = 0;
    map_msg.info.origin.orientation.x = 0;
    map_msg.info.origin.orientation.y = 0;
    map_msg.info.origin.orientation.z = 0;
    map_msg.info.origin.orientation.w = 1;
    map_msg.data.resize(road_mask.cols * road_mask.rows);

    std::vector<cv::Point3f> pc_points;

    for (int i = 0; i < road_mask.rows; i++) {
        for (int j = 0; j < road_mask.cols; j++) {
            size_t index = map_msg.data.size() - (j * road_mask.cols + i);
            if (border_mask.at<uchar>(i, j) == 255) {
                map_msg.data[index] = 100;
                pc_points.emplace_back(
                    (road_mask.rows - i - 1 - origin_point.y) * 0.01,
                    (road_mask.cols - j - 1 - origin_point.x) * 0.01,
                    0);
            } else {
                map_msg.data[index] = 255 - road_mask.at<uchar>(i, j);
            }
        }
    }
    map_pub.publish(map_msg);

    sensor_msgs::PointCloud2 pc_msg;
    pc_msg.header.stamp = stamp;
    pc_msg.header.frame_id = "base_footprint";
    sensor_msgs::PointCloud2Modifier modifier(pc_msg);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(pc_points.size());
    sensor_msgs::PointCloud2Iterator<float> iter_x(pc_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(pc_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(pc_msg, "z");
    for (const auto& pt : pc_points) {
        *iter_x = pt.x;
        *iter_y = pt.y;
        *iter_z = pt.z;
        ++iter_x;
        ++iter_y;
        ++iter_z;
    }
    pc_pub.publish(pc_msg);
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

    // cv::imshow("Raw", cv_ptr->image);

    cv::Mat rectified_image;
    model.rectifyImage(cv_ptr->image, rectified_image);
    cv::imshow("Rectified", rectified_image);

    perspective_trans.showPerspectiveSelect(rectified_image);
    perspective_trans.transform(rectified_image, true);

    cv::Mat red_mask = red_detection.getRedMask(rectified_image);

    cv::Point2f buttom_point(rectified_image.cols / 2, rectified_image.rows - 1);
    cv::Point2f seed_point = perspective_trans.transform(buttom_point);
    if (seed_point.y < 0 || seed_point.y >= perspective_trans.getConfig().image_size.height ||
        seed_point.x < 0 || seed_point.x >= perspective_trans.getConfig().image_size.width) {
        seed_point = cv::Point2f(perspective_trans.getConfig().image_size.width / 2, perspective_trans.getConfig().image_size.height - 1);
    }

    cv::Mat transformed_red_mask = perspective_trans.transform(255 - red_mask);
    cv::Mat road_mask = roadExtract(255 - transformed_red_mask, seed_point);
    cv::Mat border_mask = borderExtract(road_mask, perspective_trans.getPerspectiveMatrix(), rectified_image.size());

    int key = cv::waitKey(1);
    if (key == 'p') {
        perspective_trans.pauseOrResumeSelect();
    }
    publishMap(ros::Time::now(), road_mask, border_mask, perspective_trans.getConfig().origin_point);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ground_detector_node");
    ros::NodeHandle nh;

    camera_info = std::make_shared<camera_info_manager::CameraInfoManager>(nh);
    camera_info->loadCameraInfo("package://ground_detector/config/camera_info.yaml");

    model.fromCameraInfo(camera_info->getCameraInfo());

    perspective_trans.loadConfig("package://ground_detector/config/perspective.yaml");

    map_pub = nh.advertise<nav_msgs::OccupancyGrid>("/local_map", 1);
    pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/local_map_pc", 1);
    ros::Subscriber sub = nh.subscribe("/camera/color/image_raw", 1, imageCallback);
    cv::namedWindow("Perspective Select");
    cv::setMouseCallback("Perspective Select", mouseCallback, NULL);
    ros::spin();

    return 0;
}