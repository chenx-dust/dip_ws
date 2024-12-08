#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include <resource_retriever/retriever.h>
#include <std_msgs/Int8.h>

#include "pill_detect/match.hpp"
#include "pill_detect/pill_detect.hpp"

using namespace cv;
using namespace std;
using namespace pill_detect;

constexpr int MAX_BLUE_COUNT = 10;
constexpr int MAX_GREEN_COUNT = 10;

// 初始参数
int threshold_value = 175;
int min_area = 700;
double min_aspect_ratio = 0.50;
double max_aspect_ratio = 1.0;
double max_angle_horizon = 10;
double max_angle_diff = 10.0; // 允许的最大内角度偏差（接近直角）
int pill_min_area = 10;       // 药片最小面积阈值
int total_blue_count = 0;     // 蓝色匹配框的累计数量
int total_green_count = 0;    // 绿色匹配框的累计数量
int diff_threshold = 30;
bool color_decision = false;
bool is_blue = false;

ros::Publisher pill_pub;

PillDetect pill_detect_;

// 图像回调函数
void imageCallback(const sensor_msgs::ImageConstPtr &msg, const Mat &templ_blue, const Mat &templ_green)
{
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
        return;
    }
    auto pill_type_count = pill_detect_.detect_pill_type(cv_ptr->image);

    if (pill_type_count.first != PillDetect::PillType::UNKNOWN) {
        std_msgs::Int8 pill_type_msg;
        pill_type_msg.data = static_cast<int>(pill_type_count.first);
        pill_pub.publish(pill_type_msg);
    }

    cv::waitKey(1);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pill_detect_node");
    ros::NodeHandle nh;

    // 加载模板图像
    resource_retriever::Retriever retriever;
    auto templ_blue_resource = retriever.get("package://pill_detect/resource/blue_template.jpeg");
    std::vector<uchar> buffer_blue(templ_blue_resource.data.get(), templ_blue_resource.data.get() + templ_blue_resource.size);
    Mat templ_blue = imdecode(buffer_blue, IMREAD_COLOR);
    if (templ_blue.empty()) {
        ROS_ERROR("Failed to load blue template image");
        return -1;
    }
    auto templ_green_resource = retriever.get("package://pill_detect/resource/green_template.jpeg");
    std::vector<uchar> buffer_green(templ_green_resource.data.get(), templ_green_resource.data.get() + templ_green_resource.size);
    Mat templ_green = imdecode(buffer_green, IMREAD_COLOR);
    if (templ_green.empty()) {
        ROS_ERROR("Failed to load green template image");
        return -1;
    }
    // cv::imshow("template_blue", templ_blue);
    // cv::imshow("template_green", templ_green);
    // cv::waitKey(1);
    pill_detect_.set_template(templ_blue, templ_green);
    pill_detect_.set_config(PillDetect::DetectConfig {
        .threshold_value = nh.param<int>("threshold_value", 175),
        .min_area = nh.param<int>("min_area", 700),
        .min_aspect_ratio = nh.param<double>("min_aspect_ratio", 0.50),
        .max_aspect_ratio = nh.param<double>("max_aspect_ratio", 1.0),
        .max_angle_horizon = nh.param<double>("max_angle_horizon", 10),
        .max_angle_diff = nh.param<double>("max_angle_diff", 10.0),
        .pill_min_area = nh.param<int>("pill_min_area", 10),
        .diff_threshold = nh.param<int>("diff_threshold", 30),
    });
    // 订阅图像话题
    pill_pub = nh.advertise<std_msgs::Int8>("/turn_direction", 10);
    ros::Subscriber sub = nh.subscribe<sensor_msgs::Image>("/camera/color/image_raw", 10, boost::bind(imageCallback, _1, templ_blue, templ_green));

    ros::spin();
    return 0;
}
