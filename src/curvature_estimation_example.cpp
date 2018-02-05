#include <iostream>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/integral_image_normal.h>

#include <Eigen/Geometry>

using namespace std;

typedef cv::Mat_<float> FloatImage;
typedef cv::Mat_<cv::Vec3f> Float3Image;

typedef pcl::PointCloud<pcl::PointXYZ> PointCloudType;
typedef pcl::PointXYZ PointT;

int main(int argc, char** argv){


    Eigen::Matrix3f K;
    K << 525.0, 0.0, 319.5,
            0.0, 525.0, 239.5,
            0.0,0.0,1;
    Eigen::Matrix3f invK = K.inverse();
    float raw_depth_scale = 1e-3;

    //load depth image
    cv::Mat depth_image;
    depth_image = cv::imread(argv[1],cv::IMREAD_UNCHANGED);
    int rows = depth_image.rows;
    int cols = depth_image.cols;

    //build point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    cloud->width    = cols;
    cloud->height   = rows;
    cloud->is_dense = false;
    cloud->points.resize (cloud->width * cloud->height);
    int k=0;
    for (int r=0; r<rows; r++) {
        const unsigned short* id_ptr  = depth_image.ptr<unsigned short>(r);
        for (int c=0; c<cols; c++,id_ptr++, k++) {
            unsigned short id = *id_ptr;
            float d = id * raw_depth_scale;
            Eigen::Vector3f point = invK * Eigen::Vector3f(c*d,r*d,d);
            cloud->at(k) = pcl::PointXYZ (point.x(),point.y(),point.z());
        }
    }

    //compute normals
    pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
    ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
    ne.setMaxDepthChangeFactor (0.02f);
    ne.setNormalSmoothingSize (10.0f);
    ne.setInputCloud (cloud);
    ne.compute (*normal_cloud);

    //visualize output
    Float3Image normal_image;
    normal_image.create(rows,cols);
    FloatImage curvature_image;
    curvature_image.create(rows,cols);
    for(int r=0; r<rows; ++r){
        cv::Vec3f* normal_ptr = normal_image.ptr<cv::Vec3f>(r);
        float* curvature_ptr = curvature_image.ptr<float>(r);
        for(int c=0; c<cols; ++c, ++normal_ptr, ++curvature_ptr){
            cv::Vec3f& normal = *normal_ptr;
            float& curvature = *curvature_ptr;

            pcl::Normal& pcl_normal = normal_cloud->at(c+r*cols);

            normal = cv::Vec3f (pcl_normal.normal_x,pcl_normal.normal_y,pcl_normal.normal_z);
            curvature = pcl_normal.curvature;
        }
    }
    cv::imshow("normals",normal_image);
    cv::imshow("curvature",curvature_image);
    cv::waitKey();

    return 0;
}
