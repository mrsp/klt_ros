/* 
 * Copyright 2020 Stylianos Piperakis, Foundation for Research and Technology Hellas (FORTH)
 * License: BSD
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Foundation for Research and Technology Hellas (FORTH) 
 *		 nor the names of its contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
/**
 * @brief Visual Feature Benchmarking utilyzing RGB and Depth images
 */
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/rgbd.hpp"
#include <eigen3/Eigen/Dense>
#include "teaser/registration.h"
#include <fstream>
using namespace std;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;

class klt_ros
{
    ///current image frame
    int frame;
    ///image dimensions
    int height, width;
    ///camera distorsion
    double k1, k2, k3, t1, t2;
    ///camera calibration
    double cx, cy, fx, fy;
    /// ROS nodehanlder
    ros::NodeHandle nh;
    /// ROS image transport for image callback
    image_transport::ImageTransport it;
    /// ROS image subscriber only called when RGB image is used and not DEPTH image.
    image_transport::Subscriber image_sub_;
    vector<uchar> status;
    std::string output_path;
    ///vectors to store the Harris corners of Previous Image and Current Image (when Harris Features are enabled)
    vector<cv::Point2f> currFeatures, prevFeatures;
    ///vectors to store the SIFT Features of Previous Image and Current Image (when SIFT Features are enabled)
    std::vector<cv::KeyPoint> prevKeypoints, currKeypoints;
    ///vectors to store the SIFT Descriptors of Previous Image and Current Image (when SIFT Features are enabled)
    cv::Mat prevDescr, currDescr;
    /// camera principal point
    cv::Point2d pp;
    /// Flags for first Image Callback, first Camera Info Callback, and for new image callback
    bool firstImageCb, firstCameraInfoCb, img_inc;
    // Flags for Tracking features with KLT Tracker instead of detecting new ones, checking VO initialization, and USE depth image along with RGB
    bool trackOn, voInitialized, useDepth, benchmark_3D, mm_to_meters, publish_matches;
    /// Minimum number of features for KLT Tracking
    int MIN_NUM_FEAT;
    ///placeholders for previous and current Grayscale/RGB/Depth Image
    cv::Mat currImage, prevImage, currImageRGB, prevDepthImage, currDepthImage;
    cv::Mat  R, R_2D, t_2D, t, E, cam_intrinsics; //
    ///Eigen 3D Rotation of Previous Image to Current Image Computed with Teaser (when 3D Estimation is ran)
    Eigen::MatrixXd Rot_eig, R_f;
    ///Eigen 3D Translation of Previous Image to Current Image Computed with Teaser (when 3D Estimation is ran)
    Eigen::VectorXd t_eig, t_f;
    ///Teaser optimization parameters
    teaser::RobustRegistrationSolver::Params tparams;
    ///Teaser optimization solver
    teaser::RobustRegistrationSolver *solver;

    ///ROS RGB Image Subscriber
    message_filters::Subscriber<sensor_msgs::Image> image_sub;
    ///ROS DEPTH Image Subscriber
    message_filters::Subscriber<sensor_msgs::Image> depth_sub;
    /// ROS Synchronization for RGB and DEPTH msgs
    message_filters::Synchronizer<MySyncPolicy> *ts_sync;
    /// ROS image, depth and camera info topics
    std::string image_topic, depth_topic, cam_info_topic;
    /// SIFT feature detector
    cv::Ptr<cv::Feature2D> sift;
    //odometry path from visualization on rviz
    nav_msgs::Path odomPath;
    //odometry path publisher
    ros::Publisher odom_path_pub;
    //current pose from vo
    Eigen::Affine3d curr_pose;
    //publish matches image
    ros::Publisher matches_pub;
    //ransac reprojection threshold
    double ransacReprojThreshold;
    //absolute scale of the world
    double scale;
    std::string feature_type;
public:
    /** @fn  klt_ros(ros::NodeHandle nh_);
	 *  @brief Initializes the VO Benchmarking
     *  @param nh_ ros nodehandler 
	 */
    klt_ros(ros::NodeHandle nh_);
    /** @fn is3D()
	 *  @brief return true if klt_ros finds 3d tfs.
	 */
    inline bool is3D() const
    {
        return benchmark_3D;
    }
    /** @fn isUsingDepth()
	 *  @brief return true if klt_ros uses depth.
	 */
    inline bool isUsingDepth() const
    {
        return useDepth;
    }
    
    /** @fn teaserParams2DTFEstimation()
	 *  @brief Initializes Teaser Optimization Parameters for 2D TF Estimation
	 */    
    void teaserParams2DTFEstimation();
    /** @fn teaserParams3DTFEstimation()
	 *  @brief Initializes Teaser Optimization Parameters for 3D TF Estimation
	*/
    void teaserParams3DTFEstimation();
    /** @fn bool estimateAffineTFTeaser(const Eigen::Matrix<double, 3, Eigen::Dynamic> &src, 
                                const Eigen::Matrix<double, 3, Eigen::Dynamic> &dst,
                                std::vector<cv::DMatch> &initial_matches,
                                std::vector<cv::DMatch> &good_matches);
	 *  @brief Estimates an Affine Transformation with Teaser
     *  @param src Input keypoints from Previous Image
     *  @param dst Input keypoints from Current Image
     *  @param initial_matches Initial Correspondence between src - dst computed with KNN
     *  @param good_matches  Output Correspondence between src - dst computed with Teaser
	*/
    bool estimateAffineTFTeaser(const Eigen::Matrix<double, 3, Eigen::Dynamic> &src,
                                const Eigen::Matrix<double, 3, Eigen::Dynamic> &dst,
                                std::vector<cv::DMatch> &initial_matches,
                                std::vector<cv::DMatch> &good_matches);

    /** @fn void featureTracking(cv::Mat img_1, cv::Mat img_2, std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2, std::vector<uchar> &status);
	 *  @brief Tracks keypoints points1 from img_1 to img_2 with KLT Tracker
     *  @param img_1 Input Previous Image
     *  @param img_2 Input Current Image
     *  @param points1 Input Keypoints Detected in Previous Image (img_1)
     *  @param points2 Output Keypoints tracked in Current Image (img_2) with KLT Tracker
     *  @param status  Tracking status 
     */
    void featureTracking(cv::Mat img_1, cv::Mat img_2, std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2, std::vector<uchar> &status);
    /** @fn void featureDetection(cv::Mat img_1, std::vector<cv::Point2f> &points1);
	 *  @brief Compute Harris Corners 
     *  @param img_1 Input Image
     *  @param points1 Harris Corners detected
     */
    void featureDetection(cv::Mat img_1, std::vector<cv::Point2f> &points1);

    /** @fn void trackFeatures();
	 *  @brief applies featureDetection() and featureTracking()
     */
    void trackFeatures();

    /** @fn void transform2DKeyPoints(const std::vector<cv::KeyPoint> Keypoints,
                                   std::vector<cv::KeyPoint> &Keypoints_transformed,
                                   const cv::Mat &H)
	 *  @brief transforms 2D Keypoints detected in Previous Image to Current Image with the estimated homography H
	 *  @param points 2D keypoints in Previous Image
     *  @param Keypoints_transformed output of 2D keypoints transformed to current image
     *  @param H 3x3 homography
	 */
    void transform2DKeyPoints(const std::vector<cv::KeyPoint> Keypoints,
                                                   std::vector<cv::KeyPoint> &Keypoints_transformed,
                                                   const cv::Mat &H);
    
    /** @fn void filterPoints(const std::vector<cv::KeyPoint> &Keypoints1,
                      const std::vector<cv::KeyPoint> &Keypoints2,
                      const std::vector<cv::KeyPoint> &initial_matches,
                      std::vector<cv::KeyPoint> &good_matches,
                      double threshold)

	 *  @brief calculates l2 distance between Keypoints1 and Keypoints2 given their correspondeces in initial_matches.
	 Then fills good_matches only if l2 distance is less than threshold
	 *  @param Keypoints1 First set of keypoints
     *  @param Keypoints2 Second set of keypoints
     *  @param initial_matches matches of keypoints1 to keypoints2
     *  @param threshold threshold that points consider outlier
	 */
    void filterPoints(const std::vector<cv::KeyPoint> &Keypoints1,
                      const std::vector<cv::KeyPoint> &Keypoints2,
                      const std::vector<cv::DMatch> &initial_matches,
                      std::vector<cv::DMatch> &good_matches,
                      double threshold);

    /** @fn std::vector<Eigen::Vector3d> transform3DKeyPoints(const std::vector<Eigen::Vector3d> Keypoints, Eigen::MatrixXd Rotation, Eigen::VectorXd Translation);
	 *  @brief transforms 3D Keypoints detected in Previous Image to Current Image with the estimated from Teaser 3D TF 
	 *  @param Keypoints 3D keypoints in Previous Image
     *  @param Rotation 3D Rotation from Previous Image to Current Image
     *  @param Translation 3D Translation from Previous Image to Current Image
	 */
    std::vector<Eigen::Vector3d> transform3DKeyPoints(const std::vector<Eigen::Vector3d> Keypoints, Eigen::MatrixXd Rotation, Eigen::VectorXd Translation);

    /** @fn bool estimate2Dtf(const std::vector<cv::KeyPoint> &points1, 
                           const std::vector<cv::KeyPoint> &points2,
                           const cv::Mat &descr1,
                           const cv::Mat &descr2,
                           const std::vector<cv::DMatch> &initial_matches,
                           cv::Mat &H)
     * @brief computes the 2D TF from previous image to current image using the keypoints and descriptors detected/computed
     * @param points1 keypoints in previous image
     * @param points2 keypoints in current image
     * @param descr1 descriptors in previous image
     * @param descr2 descriptors in current image
     * @param initial_matches initial matches vector
     * @param H output homography
     */
    bool estimate2Dtf(const std::vector<cv::KeyPoint> &points1,
                      const std::vector<cv::KeyPoint> &points2,
                      const cv::Mat &descr1,
                      const cv::Mat &descr2,
                      const std::vector<cv::DMatch> &initial_matches,
                      cv::Mat &H);

    /** @fn bool estimate3Dtf(const std::vector<cv::KeyPoint> &points1,
                          const std::vector<cv::KeyPoint> &points2,
                          const cv::Mat &descr1,
                          const cv::Mat &descr2,
                          std::vector<cv::DMatch> &good_matches, 
                          std::vector<cv::KeyPoint>& m_points1, 
                          std::vector<cv::KeyPoint>& m_points2, 
                          std::vector<Eigen::Vector3d>&  m_points1_3D,
                          std::vector<Eigen::Vector3d>&  m_points2_3D,
                          std::vector<Eigen::Vector3d>&  m_points1_transformed_3D)
     * @brief computes the 3D TF from previous image to current image using the keypoints and descriptors detected/computed and the corresponding depth images
     * @param points1 keypoints in previous image
     * @param points2 keypoints in current image
     * @param descr1 descriptors in previous image
     * @param descr2 descriptors in current image
     * @param good_matches output correspondence with Teaser
     * @param m_points1 output matched keypoints in previous image
     * @param m_points2 output matched keypoints in current image
     * @param m_points1_3D output matched 3D keypoints in previous image
     * @param m_points2_3D output matched 3D keypoints in current image
     * @param m_points1_transformed_3D output matched 3D keypoints transformed from previous image to current image
     */
    bool estimate3Dtf(const std::vector<cv::KeyPoint> &points1,
                      const std::vector<cv::KeyPoint> &points2,
                      const cv::Mat &descr1,
                      const cv::Mat &descr2,
                      std::vector<cv::DMatch> &good_matches,
                      std::vector<cv::KeyPoint> &m_points1,
                      std::vector<cv::KeyPoint> &m_points2,
                      std::vector<Eigen::Vector3d> &m_points1_3D,
                      std::vector<Eigen::Vector3d> &m_points2_3D,
                      std::vector<Eigen::Vector3d> &m_points1_transformed_3D);

    /** @fn bool estimate2DtfAnd3DPoints(const std::vector<cv::KeyPoint> &points1,
                          const std::vector<cv::KeyPoint> &points2,
                          const cv::Mat &prevDescr,
                          const cv::Mat &currDescr,
                          std::vector<cv::DMatch> &good_matches, 
                          std::vector<cv::KeyPoint>& m_points1, 
                          std::vector<cv::KeyPoint>& m_points2, 
                          std::vector<cv::KeyPoint> &m_points1_transformed, 
                          std::vector<Eigen::Vector3d>&  m_points1_3D,
                          std::vector<Eigen::Vector3d>&  m_points2_3D,
                          std::vector<Eigen::Vector3d>&  m_points1_transformed_3D)
     * @brief computes the 3D TF from previous image to current image using the keypoints and descriptors detected/computed and the corresponding depth images
     * @param points1 keypoints in previous image
     * @param points2 keypoints in current image
     * @param prevDescr descriptors in previous image
     * @param currDescr descriptors in current image
     * @param good_matches output correspondence with Teaser
     * @param m_points1 output matched keypoints in previous image
     * @param m_points2 output matched keypoints in current image
     * @param m_points1_transformed output matched keypoints transformed from previous image to current image
     * @param m_points1_3D output matched 3D keypoints in previous image
     * @param m_points2_3D output matched 3D keypoints in current image
     * @param m_points1_transformed_3D output matched 3D keypoints transformed from previous image to current image
     */
    bool estimate2DtfAnd3DPoints(const std::vector<cv::KeyPoint> &points1,
                                 const std::vector<cv::KeyPoint> &points2,
                                 const cv::Mat &prevDescr,
                                 const cv::Mat &currDescr,
                                 std::vector<cv::DMatch> &good_matches,
                                 std::vector<cv::KeyPoint> &m_points1,
                                 std::vector<cv::KeyPoint> &m_points2,
                                 std::vector<cv::KeyPoint> &m_points1_transformed,
                                 std::vector<Eigen::Vector3d> &m_points1_3D,
                                 std::vector<Eigen::Vector3d> &m_points2_3D,
                                 std::vector<Eigen::Vector3d> &m_points1_transformed_3D);

    /** @fn  bool estimate2DtfAnd2DPoints(const std::vector<cv::KeyPoint> &points1,
                          const std::vector<cv::KeyPoint> &points2,
                          const cv::Mat &prevDescr,
                          const cv::Mat &currDescr,
                          std::vector<cv::DMatch> &good_matches, 
                          std::vector<cv::KeyPoint>& m_points1, 
                          std::vector<cv::KeyPoint>& m_points2, 
                          std::vector<cv::KeyPoint> &m_points1_2ed)
     * @brief computes the 3D TF from previous image to current image using the keypoints and descriptors detected/computed and the corresponding depth images
     * @param points1 keypoints in previous image
     * @param points2 keypoints in current image
     * @param prevDescr descriptors in previous image
     * @param currDescr descriptors in current image
     * @param good_matches output correspondence with Teaser
     * @param m_points1 output matched keypoints in previous image. 1 to 1 correspondence with m_points2 and m_points1_transformed
     * @param m_points2 output matched keypoints in current image. 1 to 1 correspondence with m_points1 and m_points1_transformed
     * @param m_points1_transformed output matched keypoints transformed from previous image to current image. 1 to 1 correspondence with m_points1 and m_points2
     */

    bool estimate2DtfAnd2DPoints(const std::vector<cv::KeyPoint> &points1,
                                 const std::vector<cv::KeyPoint> &points2,
                                 const cv::Mat &prevDescr,
                                 const cv::Mat &currDescr,
                                 std::vector<cv::DMatch> &good_matches,
                                 std::vector<cv::KeyPoint> &m_points1,
                                 std::vector<cv::KeyPoint> &m_points2,
                                 std::vector<cv::KeyPoint> &m_points1_transformed);
    
    bool estimate3DtfSVD(std::vector<Eigen::Vector3d> &m_points1_3D,
                         std::vector<Eigen::Vector3d> &m_points2_3D);
    
    /** @fn void imageCb(const sensor_msgs::ImageConstPtr &msg)
     * @brief image RGB callback
     */
    void imageCb(const sensor_msgs::ImageConstPtr &msg);
    /** @fn void imageDepthCb(const sensor_msgs::ImageConstPtr &img_msg,const sensor_msgs::ImageConstPtr &depth_msg);
     * @brief Synchronized image RGB and Depth callback
     */
    void imageDepthCb(const sensor_msgs::ImageConstPtr &img_msg, const sensor_msgs::ImageConstPtr &depth_msg);
    /** @fn void cameraInfoCb(const sensor_msgs::CameraInfoConstPtr &msg);
     * @brief Camera Info Callback
     */
    void cameraInfoCb(const sensor_msgs::CameraInfoConstPtr &msg);
    /** @fn vo()
     * @brief computes visual odometry
     */
    void vo();
    /** @fn void plotFeatures()
     * @brief plots Harris Corners in current Image
     */
    void plotFeatures();
    /** @fn void plotTransformedKeypoints(std::vector<cv::KeyPoint> matched_currKeypoints, std::vector<cv::KeyPoint> matched_prevKeypoints_transformed)
     *  @brief plots in an image the current keypoints and the transformed ones 
     */
    void plotTransformedKeypoints(std::vector<cv::KeyPoint> matched_currKeypoints, std::vector<cv::KeyPoint> matched_prevKeypoints_transformed);
    /** @fn void show_matches(const cv::Mat &img_1,
                           const cv::Mat &img_2,
                           const std::vector<cv::KeyPoint> keypoints1,
                           const std::vector<cv::KeyPoint> keypoints2,
                           const std::vector<cv::DMatch> &good_matches);
    /** @fn void computeTransformedKeypointsError(std::vector<cv::KeyPoint> matched_currKeypoints, std::vector<cv::KeyPoint> matched_prevKeypoints_transformed)
     *  @brief plots the matches between img_1 and img_2
     */
    void show_matches(const cv::Mat &img_1,
                      const cv::Mat &img_2,
                      const std::vector<cv::KeyPoint> keypoints1,
                      const std::vector<cv::KeyPoint> keypoints2,
                      const std::vector<cv::DMatch> &good_matches);
    /** @fn void publishMatches(const cv::Mat &img_1,
                           const cv::Mat &img_2,
                           const std::vector<cv::KeyPoint> keypoints1,
                           const std::vector<cv::KeyPoint> keypoints2,
                           const std::vector<cv::DMatch> &good_matches)  
     *  @brief publish a ros message with an images displaying the matches
     */
    void publishMatches(const cv::Mat &img_1,
                           const cv::Mat &img_2,
                           const std::vector<cv::KeyPoint> keypoints1,
                           const std::vector<cv::KeyPoint> keypoints2,
                           const std::vector<cv::DMatch> &good_matches);   
    /** @fn void publishOdomPath()
     *  @brief publish alredy constructed odometry path.
     */
    void publishOdomPath();
    /** @fn void addTfToPath(const Eigen::Matrix4d &R_f, const Eigen::VectorXd &t_f)
     *  @param Matrix4d new tf     
     *  @brief add roatiaion R_f and translation t_f to odometry path for publishing later.
     */
    void addTfToPath(const Eigen::Affine3d &pose);
    /** @fn void computeTransformedKeypointsError(std::vector<cv::KeyPoint> matched_currKeypoints, std::vector<cv::KeyPoint> matched_prevKeypoints_transformed);
     *  @brief computes the pixel error of 2D detected Keypoints from current Image and 2D transformed Keypoints from previous Image
     */
    void computeTransformedKeypointsError(std::vector<cv::KeyPoint> matched_currKeypoints, std::vector<cv::KeyPoint> matched_prevKeypoints_transformed);

    /** @fn void computeTransformedKeypoints3DError(std::vector<Eigen::Vector3d> matched_currKeypoints_3D, std::vector<Eigen::Vector3d> matched_prevKeypoints_transformed_3D);
     * @brief computes the error in meters of 3D detected Keypoints from current Image and 3D transformed Keypoints from previous Image
     */
    void computeTransformedKeypoints3DError(std::vector<Eigen::Vector3d> matched_currKeypoints_3D, std::vector<Eigen::Vector3d> matched_prevKeypoints_transformed_3D);

    /** @fn void siftFeatureDetection(const cv::Mat &img_1,
                              std::vector<cv::KeyPoint> &points1,
                              cv::Mat &descriptors1,
                              cv::Mat &depth);
     *  @brief computes sift features in 2D image space using the registered depth image
     *  @param img_1 Input image
     *  @param points1 2D Keypoints detected
     *  @param descriptors1 SIFT Descriptors computed
     */
    void siftFeatureDetection(const cv::Mat &img_1,
                              std::vector<cv::KeyPoint> &points1,
                              cv::Mat &descriptors1);

    /** @fn void siftFeatureDetection(const cv::Mat &img_1,
                              std::vector<cv::KeyPoint> &points1,
                              cv::Mat &descriptors1,
                              cv::Mat &depth);
     *  @brief computes sift features in 2D image space and neglects the one with NAN depth
     *  @param img_1 Input image
     *  @param points1 2D Keypoints detected
     *  @param descriptors1 SIFT Descriptors computed
     *  @param depth Input depth image
     */
    void siftFeatureDetection(const cv::Mat &img_1,
                              std::vector<cv::KeyPoint> &points1,
                              cv::Mat &descriptors1,
                              cv::Mat &depth);
    /** @fn void knn_simple (std::vector<cv::KeyPoint> keypoints1,
                    std::vector<cv::KeyPoint> keypoints2,
                    cv::Mat des1,
                    cv::Mat des2,
                    std::vector<cv::DMatch> &good_matches);
	 *  @brief computes correspondeces between keypoints1 and keypoints2 based on descriptors1 and descriptors2 distances
	 *  @param keypoints1 2D keypoints1 in Previous Image
	 *  @param keypoints2 2D keypoints2 in Current Image
     *  @param des1 Descriptors in Previous Image
     *  @param des2 Descriptors in Current Image
     *  @param good_matches Output correspondences
    */
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

    std::vector<cv::Point2f> getPointsfromKeyPoints(const std::vector<cv::KeyPoint> Keypoints);
    double estimateAbsoluteScale(std::vector<Eigen::Vector3d> matched_prevPoints_3D, std::vector<Eigen::Vector3d> matched_currPoints_3D, 
                                 Eigen::MatrixXd Rotation, Eigen::VectorXd Translation);

};
