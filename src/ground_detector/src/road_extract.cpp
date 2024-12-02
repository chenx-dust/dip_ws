#include "ground_detector/road_extract.hpp"

cv::Mat roadExtract(const cv::Mat& cv_image)
{
    cv::Point2i seed_point(cv_image.cols / 2, cv_image.rows - 1);
    cv::Mat mask = cv::Mat(cv_image.rows + 2, cv_image.cols + 2, CV_8UC1);
    mask.setTo(cv::Scalar(255));
    cv_image.copyTo(mask(cv::Rect(1, 1, cv_image.cols, cv_image.rows)));

    cv::Mat road_mask = cv::Mat::zeros(cv_image.rows, cv_image.cols, CV_8UC1);
    cv::floodFill(road_mask, mask, seed_point, cv::Scalar(255));
    cv::Mat color_image;
    cv::cvtColor(road_mask, color_image, cv::COLOR_GRAY2BGR);
    cv::circle(color_image, seed_point, 2, cv::Scalar(0, 0, 255), -1);
    cv::imshow("Road Mask", color_image);
    return road_mask;
}

cv::Mat borderExtract(const cv::Mat& cv_image, const cv::Mat& perspective_matrix, cv::Size origin_size)
{
    cv::Mat grad_x, grad_y;
    cv::Sobel(cv_image, grad_x, CV_16S, 1, 0);
    cv::Sobel(cv_image, grad_y, CV_16S, 0, 1);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat dilated_image;
    cv::dilate(cv_image, dilated_image, kernel);
    cv::Mat perspective_image;
    cv::warpPerspective(255 - dilated_image, perspective_image, perspective_matrix, origin_size, cv::WARP_INVERSE_MAP);
    perspective_image = 255 - perspective_image;

    cv::Mat perspective_draw;
    cv::cvtColor(perspective_image, perspective_draw, cv::COLOR_GRAY2BGR);

    std::vector<cv::Point2f> edge_points;
    for (int i = 1; i < cv_image.rows - 1; i++) {
        for (int j = 1; j < cv_image.cols - 1; j++) {
            if (grad_x.at<int16_t>(i, j) != 0 || grad_y.at<int16_t>(i, j) != 0) {
                if (cv_image.at<uchar>(i, j) == 255) {
                    continue;
                }
                // cv::Vec2d perspective_pos;
                // cv::perspectiveTransform(cv::Vec2d(j, i), perspective_pos, inv_perspective_matrix);
                // if (perspective_image.at<uchar>(perspective_pos[1], perspective_pos[0]) == 0) {
                //     border_mask.at<uchar>(i, j) = 255;
                // }
                // perspective_draw.at<cv::Vec3b>(perspective_pos[1], perspective_pos[0]) = cv::Vec3b(0, 0, 255);
                edge_points.emplace_back(j, i);
            }
        }
    }
    cv::Mat edge_points_mat(edge_points);
    cv::Mat edge_points_perspective;
    cv::perspectiveTransform(edge_points_mat, edge_points_perspective, perspective_matrix.inv());
    cv::Mat border_mask = cv::Mat::zeros(cv_image.size(), CV_8UC1);
    for (int i = 0; i < edge_points_perspective.rows; i++) {
        cv::Point2f perspective_pos_f = edge_points_perspective.at<cv::Vec2f>(i);
        cv::Point2i perspective_pos(cvRound(perspective_pos_f.x), cvRound(perspective_pos_f.y));
        bool is_border = true;
        if (perspective_pos.x < 10 || perspective_pos.x >= perspective_image.cols - 10 ||
            perspective_pos.y < 10 || perspective_pos.y >= perspective_image.rows - 10) {
            continue;
        }
        for (int k = perspective_pos.y + 1; k < perspective_image.rows; k++) {
            if (perspective_image.at<uchar>(k, perspective_pos.x) == 0) {
                is_border = false;
                break;
            }
        }
        if (is_border) {
            border_mask.at<uchar>(edge_points[i]) = 255;
            perspective_draw.at<cv::Vec3b>(perspective_pos) = cv::Vec3b(0, 0, 255);
        }
    }
    cv::imshow("Perspective Draw", perspective_draw);
    cv::imshow("Border Mask", border_mask);
    return border_mask;
}

// cv::Mat borderExtract(const cv::Mat& cv_image, cv::Point2f origin_point)
// {
//     cv::Mat polar_image;
//     int r_res = cv_image.rows - origin_point.y;
//     cv::warpPolar(cv_image, polar_image, cv::Size(0, 0), origin_point, r_res, cv::INTER_LINEAR);

//     cv::Mat grad_x, grad_y;
//     cv::Sobel(cv_image, grad_x, CV_16S, 1, 0);
//     cv::Sobel(cv_image, grad_y, CV_16S, 0, 1);
//     cv::Mat abs_grad_x, abs_grad_y;

//     cv::Mat polar_draw;
//     if (cv_image.channels() == 1) {
//         cv::cvtColor(polar_image, polar_draw, cv::COLOR_GRAY2BGR);
//     } else {
//         polar_draw = polar_image.clone();
//     }
//     cv::Mat border_mask = cv::Mat::zeros(cv_image.rows, cv_image.cols, CV_8UC1);
//     for (int i = 0; i < cv_image.rows; i++) {
//         for (int j = 0; j < cv_image.cols; j++) {
//             if (grad_x.at<int16_t>(i, j) != 0 || grad_y.at<int16_t>(i, j) != 0) {
//                 // border_mask.at<uchar>(i, j) = 255;
//                 cv::Point2f vector = cv::Point2f(j, i) - origin_point;
//                 cv::Point2i polar_pos(
//                     cvRound(cv::norm(vector)),
//                     cvRound(std::atan2(vector.y, vector.x) * r_res / 2)
//                 );
//                 // cv::circle(polar_draw, polar_pos, 2, cv::Scalar(0, 0, 255), -1);
//                 // border_mask.at<uchar>(i, j) = 255;
//                 // bool is_border = true;
//                 // for (int k = polar_pos.x + 1; k < polar_image.cols; k++) {
//                 //     polar_draw.at<cv::Vec3b>(polar_pos.y, k) = cv::Vec3b(0, 255, 0);
//                 //     if (polar_image.at<uchar>(polar_pos.y, k) == 0) {
//                 //         is_border = false;
//                 //         polar_draw.at<cv::Vec3b>(polar_pos.y, k) = cv::Vec3b(255, 255, 0);
//                 //         break;
//                 //     }
//                 // }
//                 // if (is_border) {
//                 if (polar_image.at<uchar>(polar_pos.y, polar_pos.x + 1) == 255) {
//                     polar_draw.at<cv::Vec3b>(polar_pos) = cv::Vec3b(0, 0, 255);
//                     border_mask.at<uchar>(i, j) = 255;
//                 }
//             }
//         }
//     }
//     cv::imshow("Polar Draw", polar_draw);
//     cv::imshow("Border Mask", border_mask);
//     return border_mask;
// }