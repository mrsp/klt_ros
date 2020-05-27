#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <eigen3/Eigen/Dense>
#include "teaser/registration.h"
using namespace std;

class klt_ros
{
    int frame;
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    vector<uchar> status;
    vector<cv::Point2f> currFeatures, prevFeatures; //vectors to store the coordinates of the feature points
    
    std::vector<cv::KeyPoint> prevKeypoints, currKeypoints;
    cv::Mat prevDescr,currDescr;
    
    double focal;
    cv::Point2d pp;
    bool firstImageCb, img_inc;
    bool trackOn, voInitialized;
    int MIN_NUM_FEAT;
    cv::Mat currImage, prevImage;
    cv::Mat R_f, t_f, R, t, E;
    teaser::RobustRegistrationSolver::Params tparams;
    teaser::RobustRegistrationSolver *solver;

    cv::Ptr<cv::Feature2D> sift;
    
public:
    klt_ros(ros::NodeHandle it_);

    void initTeaserParams();
  
    bool estimateAffineTFTeaser(const Eigen::Matrix<double, 3, Eigen::Dynamic> &src, 
                                const Eigen::Matrix<double, 3, Eigen::Dynamic> &dst,
                                std::vector<cv::DMatch> &initial_matches,
                                std::vector<cv::DMatch> &good_matches);
    
    void featureTracking(cv::Mat img_1, cv::Mat img_2, std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2, std::vector<uchar> &status);

    void featureDetection(cv::Mat img_1, std::vector<cv::Point2f> &points1);
    void trackFeatures();

    //void compute2Dtf(std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2);
    bool compute2Dtf(const std::vector<cv::KeyPoint> &points1,
                          const std::vector<cv::KeyPoint> &points2,
                          const cv::Mat &prevDescr,
                          const cv::Mat &currDescr,
                          std::vector<cv::DMatch> &good_matches);
    
    void imageCb(const sensor_msgs::ImageConstPtr &msg);
    void vo();
    void plotFeatures();
    
    void show_matches(const cv::Mat &img_1,
                           const cv::Mat &img_2,
                           const std::vector<cv::KeyPoint> keypoints1,
                           const std::vector<cv::KeyPoint> keypoints2,
                           const std::vector<cv::DMatch> &good_matches);
    
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