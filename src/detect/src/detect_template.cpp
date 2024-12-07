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

using namespace cv;
using namespace std;

enum PillType
{
    UNKNOWN = 0,
    BLUE = 1,
    GREEN = 2
};

// 初始参数
int threshold_value = 175;
int min_area = 500;
double min_aspect_ratio = 0.50;
double max_aspect_ratio = 1.0;
double max_angle_horizon = 10;
double max_angle_diff = 10.0; // 允许的最大内角度偏差（接近直角）
int pill_min_area = 10;       // 药片最小面积阈值
int total_blue_count = 0;     // 蓝色匹配框的累计数量
int total_green_count = 0;    // 绿色匹配框的累计数量
bool color_decision = false;
bool is_blue = false;

ros::Publisher pill_pub;

// 计算向量的角度（弧度）
double angleBetweenVectors(const cv::Point &p1, const cv::Point &p2)
{
    double dot = p1.x * p2.x + p1.y * p2.y;
    double cross = p1.x * p2.y - p1.y * p2.x;
    double angle = atan2(cross, dot); // atan2给出的是弧度，范围 [-π, π]
    return fabs(angle);
}

// 自定义比较函数：根据得分（第一个元素）对std::pair排序
bool comparePairs(const pair<float, Rect> &a, const pair<float, Rect> &b)
{
    return a.first > b.first; // 按得分从大到小排序
}

// 非极大值抑制
void nonMaxSuppression(vector<Rect> &boxes, vector<float> &scores, float overlapThresh)
{
    vector<int> picked;

    // 按照得分从高到低排序
    vector<pair<float, int>> scoreIndex;
    for (int i = 0; i < scores.size(); ++i)
    {
        scoreIndex.push_back(make_pair(scores[i], i));
    }
    sort(scoreIndex.begin(), scoreIndex.end(), greater<pair<float, int>>());

    vector<bool> suppressed(boxes.size(), false);

    for (int i = 0; i < scoreIndex.size(); ++i)
    {
        int idx = scoreIndex[i].second;
        if (suppressed[idx])
            continue;

        // 把该框标记为非抑制
        picked.push_back(idx);

        // 对每个后续框，计算其与当前框的重叠度
        for (int j = i + 1; j < scoreIndex.size(); ++j)
        {
            int idx2 = scoreIndex[j].second;
            if (suppressed[idx2])
                continue;

            float intersectionArea = (boxes[idx] & boxes[idx2]).area();                  // 两个矩形的交集区域
            float unionArea = boxes[idx].area() + boxes[idx2].area() - intersectionArea; // 联合区域

            float overlap = intersectionArea / unionArea;
            if (overlap > overlapThresh)
            {
                suppressed[idx2] = true;
            }
        }
    }

    // 输出最终保留下来的框
    vector<Rect> finalBoxes;
    for (int idx : picked)
    {
        finalBoxes.push_back(boxes[idx]);
    }

    boxes = finalBoxes;
}

// 多尺度模板匹配函数
void multiScaleTemplateMatching(const Mat &img, const Mat &templ, vector<Rect> &boxes, vector<float> &scores, double scaleFactor = 1.1, int minScale = 0, int maxScale = 3, double threshold = 0.8, float overlapThresh = 0.01, int maxMatches = 6)
{
    // 获取图像的尺寸
    int img_width = img.cols;
    int img_height = img.rows;
    int templ_width = templ.cols;
    int templ_height = templ.rows;
    // std::cout << "img" << img_width << ", " << img_height << std::endl;

    // 计算最大缩放比例（确保模板不会超过图像尺寸）
    double maxScaleRatioWidth = static_cast<double>(img_width) / templ_width;
    double maxScaleRatioHeight = static_cast<double>(img_height) / templ_height;
    double maxScaleRatio = std::min(maxScaleRatioWidth, maxScaleRatioHeight); // 获取最小的限制比例
                                                                              // 遍历不同的尺度
    for (int scale = minScale; scale <= maxScale; ++scale)
    {
        // 计算缩放比例
        double scaleRatio = pow(scaleFactor, scale);

        // 如果计算出的缩放比例超过最大比例，则限制为最大比例
        if (scaleRatio > maxScaleRatio)
        {
            scaleRatio = maxScaleRatio;
        }
        // 在每个尺度上缩放模板
        Mat scaledTemplate;
        resize(templ, scaledTemplate, Size(), scaleRatio, scaleRatio);

        // 在图像上进行模板匹配
        Mat result;
        matchTemplate(img, scaledTemplate, result, TM_CCOEFF_NORMED);

        // 获取匹配结果的尺寸
        int resultRows = result.rows;
        int resultCols = result.cols;

        // 阈值设置：得分大于阈值即认为是匹配成功
        // 遍历匹配结果矩阵，找出所有得分超过阈值的位置
        for (int i = 0; i < resultRows; ++i)
        {
            for (int j = 0; j < resultCols; ++j)
            {
                if (result.at<float>(i, j) >= threshold)
                {
                    // 找到匹配区域，记录匹配框和得分
                    Rect matchRect(j, i, scaledTemplate.cols, scaledTemplate.rows);
                    boxes.push_back(matchRect);
                    scores.push_back(result.at<float>(i, j));
                }
            }
        }
    }

    // 使用非极大值抑制去除重叠的匹配框
    nonMaxSuppression(boxes, scores, overlapThresh); // 设置重叠阈值

    // 如果匹配框多于指定数量，选择得分最高的匹配框
    if (boxes.size() > maxMatches)
    {
        vector<pair<float, Rect>> scoreBoxPairs;
        for (size_t i = 0; i < boxes.size(); ++i)
        {
            scoreBoxPairs.push_back(make_pair(scores[i], boxes[i]));
        }

        // 按得分排序，选择得分最高的匹配框
        sort(scoreBoxPairs.begin(), scoreBoxPairs.end(), comparePairs);
        scoreBoxPairs.resize(maxMatches);

        // 将选中的框和得分提取出来
        boxes.clear();
        scores.clear();
        for (const auto &pair : scoreBoxPairs)
        {
            boxes.push_back(pair.second);
            scores.push_back(pair.first);
        }
    }
}

// 图像回调函数
void imageCallback(const sensor_msgs::ImageConstPtr &msg, const Mat &templ_blue, const Mat &templ_green)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
        return;
    }

    // 转换为灰度图
    cv::Mat gray;
    cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);

    // 应用二值化
    cv::Mat binary;
    cv::threshold(gray, binary, threshold_value, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); i++)
    {
        if (hierarchy[i][3] != -1)
        {
            drawContours(binary, contours, i, Scalar(255, 255, 255), -1);
        }
    }

    // 这里使用一个宽度较大的水平矩形作为腐蚀的结构元素
    Mat element_erode = getStructuringElement(MORPH_RECT, Size(1, 11));

    // 5. 腐蚀操作，去掉水平横杠
    Mat eroded;
    cv::erode(binary, eroded, element_erode);
    // cv::imshow("erode", eroded);

    binary = eroded;

    // 创建一个 3x3 的结构元素（kernel），通常是一个矩形或者圆形
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::erode(binary, binary, element, cv::Point(-1, -1), 3);

    // // 进行开运算：先腐蚀再膨胀
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, element);

    cv::imshow("binary", binary);
    // 检测轮廓
    cv::findContours(binary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> boxes_blue;  // 存储匹配的框
    std::vector<float> scores_blue;    // 存储匹配的得分
    std::vector<cv::Rect> boxes_green; // 存储匹配的框
    std::vector<float> scores_green;   // 存储匹配的得分

    cv::Rect rectangleROI; // 存储长方形的ROI区域
    // 筛选出近似长方形的轮廓并进行模板匹配
    for (const auto &contour : contours)
    {
        // 计算轮廓的近似多边形
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, cv::arcLength(contour, true) * 0.02, true);

        // 判断是否为长方形
        if (approx.size() == 4 && cv::isContourConvex(approx))
        {
            // 计算轮廓的面积
            double area = cv::contourArea(approx);
            if (area > min_area) // 设置面积阈值
            {
                // 计算长宽比
                cv::Rect rect = cv::boundingRect(approx);
                double aspectRatio = static_cast<double>(rect.width) / rect.height;
                if (aspectRatio > min_aspect_ratio && aspectRatio < max_aspect_ratio) // 设置长宽比范围
                {
                    // 计算长方形的角度（最小矩形的旋转角度）
                    cv::RotatedRect rotatedRect = cv::minAreaRect(approx);
                    double angle = rotatedRect.angle;

                    // 如果角度接近 0（即接近垂直）
                    if (angle < -45)
                        angle += 90; // 处理负角度
                    if (angle > max_angle_horizon)
                    {
                        continue; // 如果角度不符合要求，跳过这个轮廓
                    }
                    // 计算四个角的内角
                    bool isRectangle = true;
                    for (int i = 0; i < 4; i++)
                    {
                        // 获取三个相邻顶点
                        cv::Point p1 = approx[i];
                        cv::Point p2 = approx[(i + 1) % 4];
                        cv::Point p3 = approx[(i + 2) % 4];

                        // 计算向量
                        cv::Point v1 = p1 - p2;
                        cv::Point v2 = p3 - p2;

                        // 计算夹角
                        double angle = angleBetweenVectors(v1, v2);
                        double angle_deg = angle * 180.0 / CV_PI; // 转为度数

                        // 判断角度是否接近90度
                        if (fabs(angle_deg - 90) > max_angle_diff)
                        {
                            isRectangle = false;
                            break;
                        }
                    }
                    if (isRectangle)
                    {
                        std_msgs::Int8 pill_type;
                        pill_type.data = UNKNOWN;
                        // 获取图像的尺寸
                        int imageWidth = cv_ptr->image.cols;
                        int imageHeight = cv_ptr->image.rows;

                        // 存储长方形的ROI区域
                        rectangleROI = cv::boundingRect(approx);

                        // 扩大矩形的宽度和高度
                        double newWidth = rectangleROI.width * 1.25;
                        double newHeight = rectangleROI.height * 1.25;

                        // 计算扩大的矩形的左上角位置
                        double x = rectangleROI.x - (newWidth - rectangleROI.width) / 2;
                        double y = rectangleROI.y - (newHeight - rectangleROI.height) / 2;

                        // 确保矩形左上角的坐标不会超出图像边界
                        x = std::max(0.0, std::min(x, static_cast<double>(imageWidth) - newWidth));
                        y = std::max(0.0, std::min(y, static_cast<double>(imageHeight) - newHeight));

                        // 创建新的扩大后的矩形 (cv::Rect2d)
                        cv::Rect2d expandedRectangle(x, y, newWidth, newHeight);

                        // 使用扩大的矩形作为新的ROI
                        rectangleROI = expandedRectangle;

                        cv::rectangle(cv_ptr->image, rectangleROI, cv::Scalar(0, 255, 255), 2);
                        // 提取白色长方形区域
                        cv::Mat roi = cv_ptr->image(rectangleROI);
                        // 进行模板匹配
                        multiScaleTemplateMatching(roi, templ_blue, boxes_blue, scores_blue, 1.1, -9, 3, 0.8, 0.1, 6);
                        multiScaleTemplateMatching(roi, templ_green, boxes_green, scores_green, 1.1, -9, 3, 0.8, 0.1, 8);

                        if (!color_decision)
                        {

                            // 累加每次识别到的数量
                            total_blue_count += boxes_blue.size();
                            total_green_count += boxes_green.size();
                            // std::cout << "blue_n" << total_blue_count << std::endl;
                            // std::cout << "green_n" << total_green_count << std::endl;
                            color_decision = abs(total_blue_count - total_green_count) > 15;
                        }
                        // 判断数量差异
                        if (color_decision)
                        {
                            ROS_INFO_ONCE("The color is confirmed!");
                            if (total_blue_count != 0)
                            {
                                is_blue = total_blue_count > total_green_count;
                            }
                            total_blue_count = 0;
                            total_green_count = 0;
                            if (is_blue)
                            {
                                // 如果蓝色模板的总数量较多，选择蓝色模板
                                for (size_t i = 0; i < boxes_blue.size(); ++i)
                                {
                                    boxes_blue[i].x += rectangleROI.x;
                                    boxes_blue[i].y += rectangleROI.y;
                                }
                                // 绘制蓝色匹配框
                                for (size_t i = 0; i < boxes_blue.size(); ++i)
                                {
                                    cv::rectangle(cv_ptr->image, boxes_blue[i], cv::Scalar(255, 0, 0), 2);
                                }
                                // std::cout << "蓝色个数：" << boxes_blue.size() << std::endl;
                                pill_type.data = BLUE;
                            }
                            else
                            {
                                // 如果绿色模板的总数量较多，选择绿色模板
                                for (size_t i = 0; i < boxes_green.size(); ++i)
                                {
                                    boxes_green[i].x += rectangleROI.x;
                                    boxes_green[i].y += rectangleROI.y;
                                }
                                // 绘制绿色匹配框
                                for (size_t i = 0; i < boxes_green.size(); ++i)
                                {
                                    cv::rectangle(cv_ptr->image, boxes_green[i], cv::Scalar(0, 255, 0), 2);
                                }
                                // std::cout << "绿色个数：" << boxes_green.size() << std::endl;
                                pill_type.data = GREEN;
                            }
                        }
                        else
                        {
                            // 如果蓝色和绿色的总数差异不超过 15，则继续计算平均得分并选择得分较高的模板
                            float avg_score_blue = 0.0;
                            if (!scores_blue.empty())
                            {
                                avg_score_blue = std::accumulate(scores_blue.begin(), scores_blue.end(), 0.0f) / scores_blue.size();
                            }

                            float avg_score_green = 0.0;
                            if (!scores_green.empty())
                            {
                                avg_score_green = std::accumulate(scores_green.begin(), scores_green.end(), 0.0f) / scores_green.size();
                            }

                            if (!scores_blue.empty() && (scores_green.empty() || avg_score_blue > avg_score_green))
                            {
                                // 选择蓝色模板
                                for (size_t i = 0; i < boxes_blue.size(); ++i)
                                {
                                    boxes_blue[i].x += rectangleROI.x;
                                    boxes_blue[i].y += rectangleROI.y;
                                }
                                // 绘制蓝色匹配框
                                for (size_t i = 0; i < boxes_blue.size(); ++i)
                                {
                                    cv::rectangle(cv_ptr->image, boxes_blue[i], cv::Scalar(255, 0, 0), 2);
                                }
                                // std::cout << "蓝色个数：" << boxes_blue.size() << std::endl;
                                pill_type.data = BLUE;
                            }
                            else if (!scores_green.empty())
                            {
                                // 选择绿色模板
                                for (size_t i = 0; i < boxes_green.size(); ++i)
                                {
                                    boxes_green[i].x += rectangleROI.x;
                                    boxes_green[i].y += rectangleROI.y;
                                }
                                // 绘制绿色匹配框
                                for (size_t i = 0; i < boxes_green.size(); ++i)
                                {
                                    cv::rectangle(cv_ptr->image, boxes_green[i], cv::Scalar(0, 255, 0), 2);
                                }
                                // std::cout << "绿色个数：" << boxes_green.size() << std::endl;
                                pill_type.data = GREEN;
                            }
                        }
                        pill_pub.publish(pill_type);
                    }
                }
            }
        }
    }

    cv::imshow("Detected Rectangles", cv_ptr->image);
    cv::waitKey(1); // 等待按键以刷新显示
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "template_matching_node");
    ros::NodeHandle nh;

    // 加载模板图像
    resource_retriever::Retriever retriever;
    auto templ_blue_resource = retriever.get("package://detect/resource/blue_template.jpeg");
    std::vector<uchar> buffer_blue(templ_blue_resource.data.get(), templ_blue_resource.data.get() + templ_blue_resource.size);
    Mat templ_blue = imdecode(buffer_blue, IMREAD_COLOR);
    if (templ_blue.empty())
    {
        ROS_ERROR("Failed to load blue template image");
        return -1;
    }
    auto templ_green_resource = retriever.get("package://detect/resource/green_template.jpeg");
    std::vector<uchar> buffer_green(templ_green_resource.data.get(), templ_green_resource.data.get() + templ_green_resource.size);
    Mat templ_green = imdecode(buffer_green, IMREAD_COLOR);
    if (templ_green.empty())
    {
        ROS_ERROR("Failed to load green template image");
        return -1;
    }
    cv::imshow("template_blue", templ_blue);
    cv::imshow("template_green", templ_green);
    cv::waitKey(1);
    // 订阅图像话题
    pill_pub = nh.advertise<std_msgs::Int8>("/turn_direction", 10);
    ros::Subscriber sub = nh.subscribe<sensor_msgs::Image>("/camera/color/image_raw", 10, boost::bind(imageCallback, _1, templ_blue, templ_green));

    ros::spin();
    return 0;
}
