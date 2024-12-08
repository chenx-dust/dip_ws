#include "pill_detect/pill_detect.hpp"
#include "pill_detect/match.hpp"
#include <numeric>

using namespace cv;

namespace pill_detect {

std::pair<PillDetect::PillType, int> PillDetect::detect_pill_type(const cv::Mat& img)
{
    cv::Mat draw_image = img.clone();

    // 转换为灰度图
    cv::Mat gray;
    cv::cvtColor(draw_image, gray, cv::COLOR_BGR2GRAY);

    // 应用二值化
    cv::Mat binary;
    cv::threshold(gray, binary, config_.threshold_value, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours0;
    std::vector<cv::Vec4i> hierarchy0;
    cv::findContours(binary, contours0, hierarchy0, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    assert(contours0.size() == hierarchy0.size());
    for (int i = 0; i < contours0.size(); i++) {
        if (hierarchy0[i][3] != -1) {
            drawContours(binary, contours0, i, Scalar(255, 255, 255), -1);
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
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::erode(binary, binary, element, cv::Point(-1, -1), 3);

    // // 进行开运算：先腐蚀再膨胀
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, element);

    cv::imshow("binary", binary);
    // 检测轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int max_blue_count = 0;
    int max_green_count = 0;

    cv::Rect rectangleROI; // 存储长方形的ROI区域
    // 筛选出近似长方形的轮廓并进行模板匹配
    for (const auto& contour : contours) {
        std::vector<cv::Rect> boxes_blue; // 存储匹配的框
        std::vector<float> scores_blue; // 存储匹配的得分
        std::vector<cv::Rect> boxes_green; // 存储匹配的框
        std::vector<float> scores_green; // 存储匹配的得分
        // 计算轮廓的近似多边形
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, cv::arcLength(contour, true) * 0.02, true);

        // 判断是否为长方形
        if (approx.size() != 4 || !cv::isContourConvex(approx))
            continue;
        // 计算轮廓的面积
        double area = cv::contourArea(approx);
        if (area < config_.min_area) // 设置面积阈值
            continue;
        // 计算长宽比
        cv::Rect rect = cv::boundingRect(approx);
        double aspectRatio = static_cast<double>(rect.width) / rect.height;
        if (aspectRatio < config_.min_aspect_ratio || aspectRatio > config_.max_aspect_ratio) // 设置长宽比范围
            continue;
        // 计算长方形的角度（最小矩形的旋转角度）
        cv::RotatedRect rotatedRect = cv::minAreaRect(approx);
        double angle = rotatedRect.angle;

        // 如果角度接近 0（即接近垂直）
        if (angle < -45)
            angle += 90; // 处理负角度
        if (angle > config_.max_angle_horizon)
            continue; // 如果角度不符合要求，跳过这个轮廓
                // 计算四个角的内角
        bool isRectangle = true;
        for (int i = 0; i < 4; i++) {
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
            if (fabs(angle_deg - 90) > config_.max_angle_diff) {
                isRectangle = false;
                break;
            }
        }
        if (!isRectangle)
            continue;

        PillType pill_type = PillType::UNKNOWN;
        // 获取图像的尺寸
        int imageWidth = draw_image.cols;
        int imageHeight = draw_image.rows;

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

        cv::rectangle(draw_image, rectangleROI, cv::Scalar(0, 255, 255), 2);
        // 提取白色长方形区域
        cv::Mat roi = draw_image(rectangleROI);
        // 进行模板匹配
        multiScaleTemplateMatching(roi, templ_blue_, boxes_blue, scores_blue, 1.1, -9, 3, 0.8, 0.1, MAX_BLUE_COUNT);
        multiScaleTemplateMatching(roi, templ_green_, boxes_green, scores_green, 1.1, -9, 3, 0.8, 0.1, MAX_GREEN_COUNT);

        if (!color_decision) {
            // 累加每次识别到的数量
            total_blue_count_ += boxes_blue.size();
            total_green_count_ += boxes_green.size();
            // std::cout << "blue_n" << total_blue_count << std::endl;
            // std::cout << "green_n" << total_green_count << std::endl;
            color_decision = abs(total_blue_count_ - total_green_count_) > config_.diff_threshold;
        }
        // 判断数量差异
        if (color_decision) {
            // ROS_INFO_ONCE("The color is confirmed!");
            if (total_blue_count_ != 0) {
                is_blue = total_blue_count_ > total_green_count_;
            }
            total_blue_count_ = 0;
            total_green_count_ = 0;
            if (is_blue) {
                // 如果蓝色模板的总数量较多，选择蓝色模板
                for (size_t i = 0; i < boxes_blue.size(); ++i) {
                    boxes_blue[i].x += rectangleROI.x;
                    boxes_blue[i].y += rectangleROI.y;
                }
                // 绘制蓝色匹配框
                for (size_t i = 0; i < boxes_blue.size(); ++i) {
                    cv::rectangle(draw_image, boxes_blue[i], cv::Scalar(255, 0, 0), 2);
                }
                // std::cout << "蓝色个数：" << boxes_blue.size() << std::endl;
                max_blue_count = std::max(max_blue_count, static_cast<int>(boxes_blue.size()));
            } else {
                // 如果绿色模板的总数量较多，选择绿色模板
                for (size_t i = 0; i < boxes_green.size(); ++i) {
                    boxes_green[i].x += rectangleROI.x;
                    boxes_green[i].y += rectangleROI.y;
                }
                // 绘制绿色匹配框
                for (size_t i = 0; i < boxes_green.size(); ++i) {
                    cv::rectangle(draw_image, boxes_green[i], cv::Scalar(0, 255, 0), 2);
                }
                // std::cout << "绿色个数：" << boxes_green.size() << std::endl;
                max_green_count = std::max(max_green_count, static_cast<int>(boxes_green.size()));
            }
        } else {
            // 如果蓝色和绿色的总数差异不超过 15，则继续计算平均得分并选择得分较高的模板
            float avg_score_blue = 0.0;
            if (!scores_blue.empty()) {
                avg_score_blue = std::accumulate(scores_blue.begin(), scores_blue.end(), 0.0f) / scores_blue.size();
            }

            float avg_score_green = 0.0;
            if (!scores_green.empty()) {
                avg_score_green = std::accumulate(scores_green.begin(), scores_green.end(), 0.0f) / scores_green.size();
            }

            if (!scores_blue.empty() && (scores_green.empty() || avg_score_blue > avg_score_green)) {
                // 选择蓝色模板
                for (size_t i = 0; i < boxes_blue.size(); ++i) {
                    boxes_blue[i].x += rectangleROI.x;
                    boxes_blue[i].y += rectangleROI.y;
                }
                // 绘制蓝色匹配框
                for (size_t i = 0; i < boxes_blue.size(); ++i) {
                    cv::rectangle(draw_image, boxes_blue[i], cv::Scalar(255, 0, 0), 2);
                }
                // std::cout << "蓝色个数：" << boxes_blue.size() << std::endl;
                max_blue_count = std::max(max_blue_count, static_cast<int>(boxes_blue.size()));
            } else if (!scores_green.empty()) {
                // 选择绿色模板
                for (size_t i = 0; i < boxes_green.size(); ++i) {
                    boxes_green[i].x += rectangleROI.x;
                    boxes_green[i].y += rectangleROI.y;
                }
                // 绘制绿色匹配框
                for (size_t i = 0; i < boxes_green.size(); ++i) {
                    cv::rectangle(draw_image, boxes_green[i], cv::Scalar(0, 255, 0), 2);
                }
                // std::cout << "绿色个数：" << boxes_green.size() << std::endl;
                max_green_count = std::max(max_green_count, static_cast<int>(boxes_green.size()));
            }
        }
    }

    PillType pill_type = PillType::UNKNOWN;
    int pill_count = 0;
    if (color_decision) {
        cv::namedWindow("pill_type", cv::WINDOW_NORMAL);
        cv::Mat pill_type_image(640, 640, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::setWindowProperty("pill_type", cv::WND_PROP_TOPMOST, 1);
        if (is_blue) {
            // pill_pub.publish(BLUE);
            pill_type = PillType::BLUE;
            std::cout << "蓝色个数：" << max_blue_count << std::endl;
            pill_count = max_blue_count;

            cv::putText(pill_type_image, "BLUE", cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(255, 0, 0), 5);
            cv::putText(pill_type_image, "Count: " + std::to_string(max_blue_count), cv::Point(10, 200), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 0), 5);
        } else {
            // pill_pub.publish(GREEN);
            pill_type = PillType::GREEN;
            std::cout << "绿色个数：" << max_green_count << std::endl;
            pill_count = max_green_count;

            cv::putText(pill_type_image, "GREEN", cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 255, 0), 5);
            cv::putText(pill_type_image, "Count: " + std::to_string(max_green_count), cv::Point(10, 200), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 0), 5);
        }
        cv::imshow("pill_type", pill_type_image);
    }

    cv::imshow("Detected Rectangles", draw_image);
    cv::waitKey(1); // 等待按键以刷新显示
    return std::make_pair(pill_type, pill_count);
}

void PillDetect::set_template(const cv::Mat& templ_blue, const cv::Mat& templ_green)
{
    templ_blue_ = templ_blue.clone();
    templ_green_ = templ_green.clone();
}

void PillDetect::set_config(const DetectConfig& config)
{
    config_ = config;
}
}
