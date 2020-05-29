#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

#include <eigen3/Eigen/Dense>
#include "teaser/registration.h"
#include <fstream>
using namespace std;

typedef sync_policies::ApproximateTime<sensor_msgs::Image,sensor_msgs::Image> MySyncPolicy;

class klt_ros
{
    int frame;
    int height, width;
    double k1,k2,k3,t1,t2;
    double cx, cy, fx, fy;
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    vector<uchar> status;
    vector<cv::Point2f> currFeatures, prevFeatures; //vectors to store the coordinates of the feature points
    
    std::vector<cv::KeyPoint> prevKeypoints, currKeypoints;
    cv::Mat prevDescr,currDescr;
    
    cv::Point2d pp;
    bool firstImageCb, firstCameraInfoCb, img_inc;
    bool trackOn, voInitialized, useDepth;
    int MIN_NUM_FEAT;
    cv::Mat currImage, prevImage, currImageRGB, prevDepthImage,currDepthImage;
    cv::Mat R_f, t_f, R, R_2D, t_2D, t, E;
    teaser::RobustRegistrationSolver::Params tparams;
    teaser::RobustRegistrationSolver *solver;
    

    message_filters::Subscriber<sensor_msgs::Image> *image_sub;
    message_filters::Subscriber<sensor_msgs::Image> *depth_sub;
    Synchronizer<MySyncPolicy> *ts_sync;
    std::string image_topic, depth_topic, cam_info_topic;
    cv::Ptr<cv::Feature2D> sift;
    
public:
    klt_ros(ros::NodeHandle it_);

    void teaserParams2DTFEstimation();
    void teaserParams3DTFEstimation();
  
    bool estimateAffineTFTeaser(const Eigen::Matrix<double, 3, Eigen::Dynamic> &src, 
                                const Eigen::Matrix<double, 3, Eigen::Dynamic> &dst,
                                std::vector<cv::DMatch> &initial_matches,
                                std::vector<cv::DMatch> &good_matches);
    
    void featureTracking(cv::Mat img_1, cv::Mat img_2, std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2, std::vector<uchar> &status);

    void featureDetection(cv::Mat img_1, std::vector<cv::Point2f> &points1);
    void trackFeatures();

    std::vector<cv::KeyPoint> transform2DKeyPoints(const std::vector<cv::KeyPoint> points, cv::Mat Rotation, cv::Mat Translation);

    //void estimate2Dtf(std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2);
    bool estimate2Dtf(const std::vector<cv::KeyPoint> &points1,
                          const std::vector<cv::KeyPoint> &points2,
                          const cv::Mat &prevDescr,
                          const cv::Mat &currDescr,
                          std::vector<cv::DMatch> &good_matches, 
                          std::vector<cv::KeyPoint>& m_points1, std::vector<cv::KeyPoint>& m_points2, std::vector<cv::KeyPoint>& m_points1_transformed, cv::Mat& m_d1, cv::Mat& m_d2);
    
    void imageCb(const sensor_msgs::ImageConstPtr &msg);
    void cameraInfoCb(const sensor_msgs::CameraInfoConstPtr &msg);
    void vo();
    void plotFeatures();
    void plotTransformedKeypoints(std::vector<cv::KeyPoint> matched_currKeypoints, std::vector<cv::KeyPoint> matched_prevKeypoints_transformed);
    void show_matches(const cv::Mat &img_1,
                           const cv::Mat &img_2,
                           const std::vector<cv::KeyPoint> keypoints1,
                           const std::vector<cv::KeyPoint> keypoints2,
                           const std::vector<cv::DMatch> &good_matches);
    void computeTransformedKeypointsError(std::vector<cv::KeyPoint> matched_currKeypoints, std::vector<cv::KeyPoint> matched_prevKeypoints_transformed);
    void siftFeatureDetection(const cv::Mat &img_1, 
                              std::vector<cv::KeyPoint> &points1,
                              cv::Mat &descriptors1);
    
    void knn_simple(std::vector<cv::KeyPoint> keypoints1,
                    std::vector<cv::KeyPoint> keypoints2,
                    cv::Mat des1,
                    cv::Mat des2,
                    std::vector<cv::DMatch> &good_matches);
    
    void knn(std::vector<cv::KeyPoint> keypoints1,
             std::vector<cv::KeyPoint> keypoints2,
             cv::Mat des1,
             cv::Mat des2,
             std::vector<cv::DMatch> &good_matches);
    
    void knn_mutual(std::vector<cv::KeyPoint> keypoints1,
                         std::vector<cv::KeyPoint> keypoints2,
                         cv::Mat des1,
                         cv::Mat des2,
                         std::vector<cv::DMatch> &good_matches);
};