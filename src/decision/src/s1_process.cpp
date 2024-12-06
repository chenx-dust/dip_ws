#include "decision/decision.hpp"

using namespace decision;

double angle_diff(double a, double b)
{
    double diff = a - b;
    if (diff > CV_PI)
        diff -= 2 * CV_PI;
    if (diff < -CV_PI)
        diff += 2 * CV_PI;
    return diff;
}

cv::Point2f project_point_to_line(cv::Point2f start, float rho, float theta)
{
    // 计算 theta 的三角函数
    float cos_theta = std::cos(theta);
    float sin_theta = std::sin(theta);

    // 点到直线的有符号距离 d
    float d = start.x * cos_theta + start.y * sin_theta - rho;

    // 计算投影点
    float x_p = start.x - d * cos_theta;
    float y_p = start.y - d * sin_theta;

    return cv::Point2f(x_p, y_p);
}

cv::Point2f Decision::find_border_infront(const cv::Mat& image, cv::Point bot_location, cv::Point bot_direction, cv::Vec2d line)
{
    double rho = line[0], theta = line[1];

    cv::Point2f direction = cv::Point2f(-sin(theta), cos(theta));
    direction /= cv::norm(direction);

    if (bot_direction.dot(direction) < 0)
        direction = -direction;

    cv::Point2f start = project_point_to_line(bot_location, rho, theta);

    cv::Point2f now = start;
    bool found = false;
    while (now.x >= 0 && now.x < image.cols && now.y >= 0 && now.y < image.rows) {
        now += direction;
        if (image.at<uchar>(now) == 255) {
            found = true;
            break;
        }
    }

    if (!found) {
        std::cout << "border not found" << std::endl;
    }

    cv::Point2f end = now - config_.front_length * direction;

    cv::Mat show;
    cv::cvtColor(image, show, cv::COLOR_GRAY2BGR);
    cv::circle(show, bot_location, 5, cv::Scalar(0, 0, 255), -1);
    cv::line(show, start, end, cv::Scalar(0, 255, 0), 2);
    cv::flip(show, show, 0);
    cv::imshow("t_show", show);

    return end;
}

cv::Vec2d Decision::find_edge(const cv::Mat& border_map, double yaw)
{
    yaw += CV_PI / 2;
    if (yaw < 0)
        yaw += CV_PI;
    if (yaw > CV_PI)
        yaw -= CV_PI;

    std::vector<cv::Vec2f> lines;
    cv::HoughLines(border_map, lines, 1, CV_PI / 180, config_.hough_threshold);

    cv::Mat show;
    cv::cvtColor(border_map, show, cv::COLOR_GRAY2BGR);

    struct TotalLine {
        double rho_total = 0;
        double theta_total = 0;
        unsigned count = 0;
    } left_line, right_line;

    bool is_left = true;

    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0];
        float theta = lines[i][1];

        cv::Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        cv::line(show, pt1, pt2, cv::Scalar(0, 0, 255), 2);

        std::cout << "theta: " << theta << " yaw: " << yaw << " diff: " << angle_diff(theta, yaw) << std::endl;

        // 去除不同方向的线
        if (abs(angle_diff(theta, yaw)) > config_.angle_threshold)
            continue;

        if (is_left) {
            if (left_line.count == 0) {
                left_line.rho_total = rho;
                left_line.theta_total = theta;
                left_line.count++;
            } else if (abs(rho - left_line.rho_total / left_line.count) < config_.rho_threshold) {
                left_line.rho_total += rho;
                left_line.theta_total += theta;
                left_line.count++;
            } else {
                is_left = false;
            }
        }
        if (!is_left) {
            if (right_line.count == 0) {
                right_line.rho_total = rho;
                right_line.theta_total = theta;
                right_line.count++;
            } else if (abs(rho - right_line.rho_total / right_line.count) < config_.rho_threshold) {
                right_line.rho_total += rho;
                right_line.theta_total += theta;
                right_line.count++;
            } else {
                break;
            }
        }
    }

    double left_rho = left_line.rho_total / left_line.count;
    double left_theta = left_line.theta_total / left_line.count;
    double right_rho = right_line.rho_total / right_line.count;
    double right_theta = right_line.theta_total / right_line.count;
    // std::cout << left_rho << " " << left_theta << " " << right_rho << " " << right_theta << std::endl;

    cv::Point pt1, pt2;
    double a = cos(left_theta), b = sin(left_theta);
    // std::cout << "a " << a << " b " << b << std::endl;
    double x0 = a * left_rho, y0 = b * left_rho;
    pt1.x = cvRound(x0 + 1000 * (-b));
    pt1.y = cvRound(y0 + 1000 * (a));
    pt2.x = cvRound(x0 - 1000 * (-b));
    pt2.y = cvRound(y0 - 1000 * (a));
    cv::line(show, pt1, pt2, cv::Scalar(0, 255, 0), 2);

    a = cos(right_theta), b = sin(right_theta);
    // std::cout << "a " << a << " b " << b << std::endl;
    x0 = a * right_rho, y0 = b * right_rho;
    pt1.x = cvRound(x0 + 1000 * (-b));
    pt1.y = cvRound(y0 + 1000 * (a));
    pt2.x = cvRound(x0 - 1000 * (-b));
    pt2.y = cvRound(y0 - 1000 * (a));
    cv::line(show, pt1, pt2, cv::Scalar(0, 255, 0), 2);

    double rho = (left_rho + right_rho) / 2;
    double theta = (left_theta + right_theta) / 2;
    std::cout << rho << " " << theta << std::endl;
    a = cos(theta), b = sin(theta);
    // std::cout << "a " << a << " b " << b << std::endl;
    x0 = a * rho, y0 = b * rho;
    pt1.x = cvRound(x0 + 1000 * (-b));
    pt1.y = cvRound(y0 + 1000 * (a));
    pt2.x = cvRound(x0 - 1000 * (-b));
    pt2.y = cvRound(y0 - 1000 * (a));
    cv::line(show, pt1, pt2, cv::Scalar(255, 255, 0), 2);

    cv::flip(show, show, 0);
    cv::imshow("show", show);

    return cv::Vec2d(rho, theta);
}
