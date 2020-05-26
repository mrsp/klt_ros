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

class klt_ros
{
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    vector<uchar> status;
    vector<cv::Point2f> currFeatures, prevFeatures; //vectors to store the coordinates of the feature points
    double focal;
    cv::Point2d pp;
    bool firstImageCb;
    bool trackOn;
    int MIN_NUM_FEAT;
    cv::Mat currImage, prevImage;
    cv::Mat R_f, t_f, R, t, E;
    teaser::RobustRegistrationSolver::Params tparams;
    teaser::RobustRegistrationSolver *solver;

public:
    klt_ros(ros::NodeHandle it_);

    void initTeaserParams();
    void estimateAffineTFTeaser(Eigen::Matrix<double, 3, Eigen::Dynamic> src, Eigen::Matrix<double, 3, Eigen::Dynamic> dst);
    void featureTracking(cv::Mat img_1, cv::Mat img_2, std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2, std::vector<uchar> &status);

    void featureDetection(cv::Mat img_1, std::vector<cv::Point2f> &points1);
    void trackFeatures();

    void compute2Dtf(std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2);
    void imageCb(const sensor_msgs::ImageConstPtr &msg);
    void vo();
    void plotFeatures();
};