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
    parameter_object->AddParameterMap(elastix::ParameterObject::GetDefaultParameterMap("affine"));
    filter->SetParameterObject(parameter_object);
    filter->SetFixedImage(itk_src);
    filter->SetMovingImage(itk_dst);
    filter->SetNumberOfThreads(8);
    filter->Update();

    auto result_transform_parameters = filter->GetTransformParameterObject()->GetParameterMaps();
    auto transform = result_transform_parameters[0]["TransformParameters"];

    cv::Mat result = cv::Mat::eye(3, 2, CV_32F);
    for (int i = 0; i < transform.size(); i++) {
        result.at<float>(i) = atof(transform[i].c_str());
    }
    std::cout << result.t() << std::endl;
    return result.t();
}
}

