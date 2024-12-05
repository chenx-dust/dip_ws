#include "map_registration/registration.hpp"

#include <elastixlib.h>
#include <elxParameterObject.h>
#include <itkElastixRegistrationMethod.h>
#include <itkOpenCVImageBridge.h>

namespace map_registration {
typedef itk::Image<unsigned char, 2> ImageType;

cv::Mat registration(const cv::Mat& src, const cv::Mat& dst)
{
    ImageType::Pointer itk_src = itk::OpenCVImageBridge::CVMatToITKImage<ImageType>(src);
    ImageType::Pointer itk_dst = itk::OpenCVImageBridge::CVMatToITKImage<ImageType>(dst);

    auto filter = itk::ElastixRegistrationMethod<ImageType, ImageType>::New();
    auto parameter_object = elastix::ParameterObject::New();
    parameter_object->AddParameterMap(elastix::ParameterObject::GetDefaultParameterMap("rigid"));
    filter->SetParameterObject(parameter_object);
    filter->SetFixedImage(itk_src);
    filter->SetMovingImage(itk_dst);
    filter->SetNumberOfThreads(8);
    filter->UpdateLargestPossibleRegion();

    auto result_transform_parameters = filter->GetTransformParameterObject()->GetParameterMaps();
    auto transform = result_transform_parameters[0]["TransformParameters"];
    auto center = result_transform_parameters[0]["CenterOfRotationPoint"];

    // cv::Mat result = cv::Mat::eye(3, 2, CV_32F);
    // for (int i = 0; i < transform.size(); i++) {
    //     result.at<float>(i) = atof(transform[i].c_str());
    // }
    // std::cout << result.t() << std::endl;
    // return result.t();

    cv::Point2f center_point(atof(center[0].c_str()), atof(center[1].c_str()));
    cv::Vec3f result_vec(atof(transform[0].c_str()), atof(transform[1].c_str()), atof(transform[2].c_str()));

    cv::Mat result_mat = cv::getRotationMatrix2D(center_point, result_vec[0] * 180 / M_PI, 1);
    cv::Vec2d translation(-result_vec[1], -result_vec[2]);
    // translation = result_mat.dot(translation);
    cv::Vec2d translation_result(
        result_mat.at<double>(0, 0) * translation[0] + result_mat.at<double>(0, 1) * translation[1],
        result_mat.at<double>(1, 0) * translation[0] + result_mat.at<double>(1, 1) * translation[1]
    );
    result_mat.at<double>(0, 2) += translation_result[0];
    result_mat.at<double>(1, 2) += translation_result[1];
// transform_mat = cv2.getRotationMatrix2D(center, angle * 180 / np.pi, 1)
// print(transform_mat)
// translation = np.array([-x, -y, 0])
// translation = transform_mat.dot(translation)
// transform_mat[0, 2] += translation[0]
// transform_mat[1, 2] += translation[1]
// print(transform_mat)
    return result_mat;
}
}

