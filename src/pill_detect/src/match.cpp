#include "pill_detect/match.hpp"

#include <cmath>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

namespace pill_detect {

double angleBetweenVectors(const cv::Point& p1, const cv::Point& p2)
{
    double dot = p1.x * p2.x + p1.y * p2.y;
    double cross = p1.x * p2.y - p1.y * p2.x;
    double angle = atan2(cross, dot); // atan2给出的是弧度，范围 [-π, π]
    return fabs(angle);
}

// 自定义比较函数：根据得分（第一个元素）对std::pair排序
bool comparePairs(const pair<float, Rect>& a, const pair<float, Rect>& b)
{
    return a.first > b.first; // 按得分从大到小排序
}

// 非极大值抑制
vector<Rect> nonMaxSuppression(const vector<Rect>& boxes, const vector<float>& scores, float overlapThresh)
{
    assert(boxes.size() == scores.size());
    vector<int> picked;

    // 按照得分从高到低排序
    vector<pair<float, int>> scoreIndex;
    for (int i = 0; i < scores.size(); ++i) {
        scoreIndex.push_back(make_pair(scores[i], i));
    }
    sort(scoreIndex.begin(), scoreIndex.end(), greater<pair<float, int>>());

    vector<bool> suppressed(boxes.size(), false);

    for (int i = 0; i < scoreIndex.size(); ++i) {
        int idx = scoreIndex[i].second;
        if (suppressed[idx])
            continue;

        // 把该框标记为非抑制
        picked.push_back(idx);

        // 对每个后续框，计算其与当前框的重叠度
        for (int j = i + 1; j < scoreIndex.size(); ++j) {
            int idx2 = scoreIndex[j].second;
            if (suppressed[idx2])
                continue;

            float intersectionArea = (boxes[idx] & boxes[idx2]).area(); // 两个矩形的交集区域
            float unionArea = boxes[idx].area() + boxes[idx2].area() - intersectionArea; // 联合区域

            float overlap = intersectionArea / unionArea;
            if (overlap > overlapThresh) {
                suppressed[idx2] = true;
            }
        }
    }

    // 输出最终保留下来的框
    vector<Rect> finalBoxes;
    for (int idx : picked) {
        finalBoxes.push_back(boxes[idx]);
    }

    return finalBoxes;
}

// 多尺度模板匹配函数
void multiScaleTemplateMatching(const Mat& img, const Mat& templ, vector<Rect>& boxes, vector<float>& scores, double scaleFactor, int minScale, int maxScale, double threshold, float overlapThresh, int maxMatches)
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
    for (int scale = minScale; scale <= maxScale; ++scale) {
        // 计算缩放比例
        double scaleRatio = pow(scaleFactor, scale);

        // 如果计算出的缩放比例超过最大比例，则限制为最大比例
        if (scaleRatio > maxScaleRatio) {
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
        for (int i = 0; i < resultRows; ++i) {
            for (int j = 0; j < resultCols; ++j) {
                if (result.at<float>(i, j) >= threshold) {
                    // 找到匹配区域，记录匹配框和得分
                    Rect matchRect(j, i, scaledTemplate.cols, scaledTemplate.rows);
                    boxes.push_back(matchRect);
                    scores.push_back(result.at<float>(i, j));
                }
            }
        }
    }

    // 使用非极大值抑制去除重叠的匹配框
    boxes = nonMaxSuppression(boxes, scores, overlapThresh); // 设置重叠阈值

    // 如果匹配框多于指定数量，选择得分最高的匹配框
    if (boxes.size() > maxMatches) {
        vector<pair<float, Rect>> scoreBoxPairs;
        for (size_t i = 0; i < boxes.size(); ++i) {
            scoreBoxPairs.push_back(make_pair(scores[i], boxes[i]));
        }

        // 按得分排序，选择得分最高的匹配框
        sort(scoreBoxPairs.begin(), scoreBoxPairs.end(), comparePairs);
        scoreBoxPairs.resize(maxMatches);

        // 将选中的框和得分提取出来
        boxes.clear();
        scores.clear();
        for (const auto& pair : scoreBoxPairs) {
            boxes.push_back(pair.second);
            scores.push_back(pair.first);
        }
    }
}

} // namespace pill_detect
