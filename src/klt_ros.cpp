#include <klt_ros/klt_ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <Eigen/Geometry> 

#define PUB_ODOM_PATH_TOPIC "/klt_ros/odomPath"
#define PUB_MATCHES_TOPIC "/klt_ros/img/matches"

#include <klt_ros/utils.h>

klt_ros::klt_ros(ros::NodeHandle nh_) : it(nh_)
{

    nh = nh_;
    sift = cv::xfeatures2d::SIFT::create();

    img_inc = false;
    firstImageCb = true;
    firstCameraInfoCb = true;
    MIN_NUM_FEAT = 3;

    trackOn = false;
    voInitialized = false;

    frame = 0;
    R_f = cv::Mat::eye(3, 3, CV_64F);
    t_f = cv::Mat::zeros(3, 1, CV_64F);
    R_2D = cv::Mat::eye(3, 3, CV_64F);
    t_2D = cv::Mat::zeros(3, 1, CV_64F);
    R = cv::Mat::eye(3, 3, CV_64F);
    t = cv::Mat::zeros(3, 1, CV_64F);
    Rot_eig.resize(3, 3);
    Rot_eig.setIdentity();
    t_eig.resize(3);
    t_eig.setZero();
    ros::NodeHandle n_p("~");

    n_p.param<std::string>("image_topic", image_topic, "/camera/rgb/image_rect_color");
    n_p.param<bool>("useDepth", useDepth, true);
    n_p.param<std::string>("depth_topic", depth_topic, "/camera/depth_registered/sw_registered/image_rect");
    n_p.param<std::string>("cam_info_topic", cam_info_topic, "/camera/rgb/camera_info");
    n_p.param<bool>("benchmark_3D", benchmark_3D, true);
    n_p.param<std::string>("output_path", output_path, "/tmp");
    n_p.param<bool>("mm_to_meters", mm_to_meters, false);
    n_p.param<bool>("publish_matches", publish_matches, false);
    
    if(publish_matches)
        matches_pub = n_p.advertise<sensor_msgs::Image>(PUB_MATCHES_TOPIC, 100);
    
    if (useDepth)
    {
        image_sub.subscribe(nh, image_topic, 1);
        depth_sub.subscribe(nh, depth_topic, 1);

        ts_sync = new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10), image_sub, depth_sub);

        ts_sync->registerCallback(boost::bind(&klt_ros::imageDepthCb, this, _1, _2));
        
        odom_path_pub = n_p.advertise<nav_msgs::Path>(PUB_ODOM_PATH_TOPIC, 50);
    }
    else
        image_sub_ = it.subscribe(image_topic, 1, &klt_ros::imageCb, this);

    ROS_INFO("Waiting camera info");
    while (ros::ok())
    {
        sensor_msgs::CameraInfoConstPtr cam_info = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(cam_info_topic);
        if (cam_info)
        {
            cameraInfoCb(cam_info);
            break;
        }
    }
}

void klt_ros::imageDepthCb(const sensor_msgs::ImageConstPtr &img_msg, const sensor_msgs::ImageConstPtr &depth_msg)
{

    ROS_INFO("Image and Depth Cb");

    cv_bridge::CvImagePtr cv_ptr;
    img_inc = true;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception &e)

    {
        ROS_ERROR("cv_bridge RGB exception: %s", e.what());
        return;
    }

    cv_bridge::CvImagePtr cv_depth_ptr;
    try
    {
        cv_depth_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    }
    catch (cv_bridge::Exception &e)

    {
        ROS_ERROR("cv_bridge DEPTH exception: %s", e.what());
        return;
    }    
    
    if(mm_to_meters)
        cv_depth_ptr->image *= 0.001;

    if (firstImageCb)
    {
        //prevImage = cv_ptr->image;
        cvtColor(cv_ptr->image, prevImage, cv::COLOR_BGR2GRAY);
        prevDepthImage = cv_depth_ptr->image;
        if (!useDepth)
        {
            siftFeatureDetection(prevImage, prevKeypoints, prevDescr, prevDepthImage);
        }
        else
        {
            siftFeatureDetection(prevImage, prevKeypoints, prevDescr);
        }

        firstImageCb = false;
    }
    else
    {
        currImageRGB = cv_ptr->image;
        cvtColor(cv_ptr->image, currImage, cv::COLOR_BGR2GRAY);
        currDepthImage = cv_depth_ptr->image;
        if (!voInitialized)
            voInitialized = true;
    }
    frame++;
}

void klt_ros::cameraInfoCb(const sensor_msgs::CameraInfoConstPtr &msg)
{

    if (firstCameraInfoCb)
    {

        height = msg->height;
        width = msg->width;

        k1 = msg->D[0];
        k2 = msg->D[1];
        t1 = msg->D[2];
        t2 = msg->D[3];
        k3 = msg->D[4];

        fx = msg->K[0];
        cx = msg->K[2];
        fy = msg->K[5];
        cy = msg->K[6];

        pp.x = cx;
        pp.y = cy;
        firstCameraInfoCb = false;
    }
}

void klt_ros::imageCb(const sensor_msgs::ImageConstPtr &msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    img_inc = true;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception &e)

    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    if (firstImageCb)
    {
        //prevImage = cv_ptr->image;
        cvtColor(cv_ptr->image, prevImage, cv::COLOR_BGR2GRAY);
        siftFeatureDetection(prevImage, prevKeypoints, prevDescr);

        firstImageCb = false;
    }
    else
    {
        currImageRGB = cv_ptr->image;
        cvtColor(cv_ptr->image, currImage, cv::COLOR_BGR2GRAY);
        if (!voInitialized)
            voInitialized = true;
    }
    frame++;
}

bool klt_ros::estimate2DtfAnd2DPoints(const std::vector<cv::KeyPoint> &points1,
                                      const std::vector<cv::KeyPoint> &points2,
                                      const cv::Mat &prevDescr,
                                      const cv::Mat &currDescr,
                                      std::vector<cv::DMatch> &good_matches,
                                      std::vector<cv::KeyPoint> &m_points1,
                                      std::vector<cv::KeyPoint> &m_points2,
                                      std::vector<cv::KeyPoint> &m_points1_transformed)
{
    m_points1.clear();
    m_points2.clear();
    m_points1_transformed.clear();

    bool matched = estimate2Dtf(points1, points2, prevDescr, currDescr, good_matches);

    if (!matched)
        return false;

    m_points1.reserve(good_matches.size());
    m_points2.reserve(good_matches.size());

    for (int i = 0; i < good_matches.size(); i++)
    {
        //prev key points
        cv::DMatch m = good_matches[i];
        int qidx = m.queryIdx;
        int tidx = m.trainIdx;

        cv::KeyPoint p1 = points1[qidx];
        cv::KeyPoint p2 = points2[tidx];
        m_points1.push_back(p1);
        m_points2.push_back(p2);
    }

    m_points1_transformed = transform2DKeyPoints(m_points1, R_2D, t_2D);

    return true;
}

bool klt_ros::estimate2DtfAnd3DPoints(const std::vector<cv::KeyPoint> &points1,
                                      const std::vector<cv::KeyPoint> &points2,
                                      const cv::Mat &prevDescr,
                                      const cv::Mat &currDescr,
                                      std::vector<cv::DMatch> &good_matches,
                                      std::vector<cv::KeyPoint> &m_points1,
                                      std::vector<cv::KeyPoint> &m_points2,
                                      std::vector<cv::KeyPoint> &m_points1_transformed,
                                      std::vector<Eigen::Vector3d> &m_points1_3D,
                                      std::vector<Eigen::Vector3d> &m_points2_3D,
                                      std::vector<Eigen::Vector3d> &m_points1_transformed_3D)
{
    m_points1.clear();
    m_points2.clear();
    m_points1_transformed.clear();

    m_points1_3D.clear();
    m_points2_3D.clear();
    m_points1_transformed_3D.clear();

    good_matches.clear();

    bool matched = estimate2Dtf(points1, points2, prevDescr, currDescr, good_matches);

    if (!matched)
        return false;

    m_points1.reserve(good_matches.size());
    m_points2.reserve(good_matches.size());

    m_points1_3D.reserve(good_matches.size());
    m_points2_3D.reserve(good_matches.size());
    //m_points1_transformed_3D.reserve(good_matches.size());

    float min_depth = 0.0001f;

    for (int i = 0; i < good_matches.size(); i++)
    {
        //prev key points
        cv::DMatch m = good_matches[i];
        int qidx = m.queryIdx;
        int tidx = m.trainIdx;

        cv::KeyPoint p1 = points1[qidx];
        cv::KeyPoint p2 = points2[tidx];

        //current key points
        int x1 = p1.pt.y + 0.5f;
        int y1 = p1.pt.x + 0.5f;
        float d1 = prevDepthImage.at<float>(x1, y1);

        int x2 = p2.pt.y + 0.5f;
        int y2 = p2.pt.x + 0.5f;
        float d2 = currDepthImage.at<float>(x2, y2);

        // some near plane constraint and NaN elimination
        if (d1 < 0.0001f || d2 < 0.0001f ||
            d1 != d1 || d2 != d2)
            continue;

        Eigen::Vector3d v1((x1 - cx) * d1 / fx,
                           (y1 - cy) * d1 / fy,
                           d1);

        Eigen::Vector3d v2((x2 - cx) * d2 / fx,
                           (y2 - cy) * d2 / fy,
                           d2);

        m_points1.push_back(p1);
        m_points2.push_back(p2);

        m_points1_3D.push_back(v1);
        m_points2_3D.push_back(v2);
    }

    if (m_points1.size() == 0)
        return false;

    m_points1_transformed = transform2DKeyPoints(m_points1, R_2D, t_2D);

    m_points1_transformed_3D.reserve(m_points1.size());
    for (int i = 0; i < m_points1.size(); i++)
    {
        cv::KeyPoint pt = m_points1_transformed[i];
        int xt = pt.pt.x + 0.5f;
        int yt = pt.pt.y + 0.5f;
        float dt = currDepthImage.at<float>(xt, yt);

        Eigen::Vector3d vt((xt - cx) * dt / fx,
                           (yt - cy) * dt / fy,
                           dt);
        m_points1_transformed_3D.push_back(vt);
    }

    return true;
}

bool klt_ros::estimate3Dtf(const std::vector<cv::KeyPoint> &points1,
                           const std::vector<cv::KeyPoint> &points2,
                           const cv::Mat &descr1,
                           const cv::Mat &descr2,
                           std::vector<cv::DMatch> &good_matches,
                           std::vector<cv::KeyPoint> &m_points1,
                           std::vector<cv::KeyPoint> &m_points2,
                           std::vector<Eigen::Vector3d> &m_points1_3D,
                           std::vector<Eigen::Vector3d> &m_points2_3D,
                           std::vector<Eigen::Vector3d> &m_points1_transformed_3D)
{

    //First Run the KNN Matcher to derive an Initial Correspondence.
    std::vector<cv::DMatch> knn_matches;
    knn_simple(points1, points2, descr1, descr2, knn_matches);

    if (knn_matches.size() < MIN_NUM_FEAT)
    {
        ROS_INFO("Not Enough Correspondences to Compute Camera Egomotion");
        return false;
    }

    m_points1.reserve(knn_matches.size());
    m_points2.reserve(knn_matches.size());
    m_points1_3D.reserve(knn_matches.size());
    m_points2_3D.reserve(knn_matches.size());
    m_points1_transformed_3D.reserve(knn_matches.size());

    float min_depth = 0.0001f;

    for (int i = 0; i < knn_matches.size(); i++)
    {
        //prev key points
        cv::DMatch m = knn_matches[i];
        int qidx = m.queryIdx;
        int tidx = m.trainIdx;

        cv::KeyPoint p1 = points1[qidx];
        cv::KeyPoint p2 = points2[tidx];

        //         //current key points
        int x1 = p1.pt.y + 0.5f;
        int y1 = p1.pt.x + 0.5f;
        float d1 = prevDepthImage.at<float>(x1, y1);

        int x2 = p2.pt.y + 0.5f;
        int y2 = p2.pt.x + 0.5f;
        float d2 = currDepthImage.at<float>(x2, y2);

        // Near plane constraint and NaN elimination

        if (d1 < 0.0001f || d2 < 0.0001f ||
            d1 != d1 || d2 != d2)
        {
            //This should never happen
            std::cout << "[WARNIN] NaN values on depth." << std::endl;
        }

        Eigen::Vector3d v1((x1 - cx) * d1 / fx,
                           (y1 - cy) * d1 / fy,
                           d1);

        Eigen::Vector3d v2((x2 - cx) * d2 / fx,
                           (y2 - cy) * d2 / fy,
                           d2);

        //m_points1.push_back(p1);
        //m_points2.push_back(p2);
        m_points1_3D.push_back(v1);
        m_points2_3D.push_back(v2);
    }

    //Prepare Keypoints for Teaser
    Eigen::Matrix<double, 3, Eigen::Dynamic> src(3, m_points1_3D.size());
    Eigen::Matrix<double, 3, Eigen::Dynamic> dst(3, m_points1_3D.size());
    for (int i = 0; i < m_points1_3D.size(); i++)
    {
        src.col(i) << m_points1_3D[i];
        dst.col(i) << m_points2_3D[i];
    }

    //Estimate the 3D Affine Transformation with Teaser
    teaserParams3DTFEstimation();

    bool matched=estimateAffineTFTeaser(src, dst, knn_matches, good_matches);
    if(!matched)
        return false;

    //Derive the Inliers as computed by Teaser
    m_points1.clear();
    m_points2.clear();
    m_points1_3D.clear();
    m_points2_3D.clear();

    m_points1.reserve(good_matches.size());
    m_points2.reserve(good_matches.size());
    m_points1_3D.reserve(good_matches.size());
    m_points2_3D.reserve(good_matches.size());
    m_points1_transformed_3D.reserve(good_matches.size());
    for (int i = 0; i < good_matches.size(); i++)
    {
        //prev key points
        cv::DMatch m = good_matches[i];
        int qidx = m.queryIdx;
        int tidx = m.trainIdx;

        cv::KeyPoint p1 = points1[qidx];
        cv::KeyPoint p2 = points2[tidx];
        //current key points
        int x1 = p1.pt.y + 0.5f;
        int y1 = p1.pt.x + 0.5f;
        float d1 = prevDepthImage.at<float>(x1, y1);

        int x2 = p2.pt.y + 0.5f;
        int y2 = p2.pt.x + 0.5f;
        float d2 = currDepthImage.at<float>(x2, y2);
        Eigen::Vector3d v1((x1 - cx) * d1 / fx,
                           (y1 - cy) * d1 / fy,
                           d1);

        Eigen::Vector3d v2((x2 - cx) * d2 / fx,
                           (y2 - cy) * d2 / fy,
                           d2);
        m_points1.push_back(p1);
        m_points2.push_back(p2);
        m_points1_3D.push_back(v1);
        m_points2_3D.push_back(v2);
    }

    //Transform the 3D points of Image 1 to the 3D Space with the Teaser Affine Transformation
    m_points1_transformed_3D = transform3DKeyPoints(m_points1_3D, Rot_eig, t_eig);
    return true;
}
std::vector<Eigen::Vector3d> klt_ros::transform3DKeyPoints(const std::vector<Eigen::Vector3d> Keypoints, Eigen::MatrixXd Rotation, Eigen::VectorXd Translation)
{    
    std::vector<Eigen::Vector3d> Keypoints_transformed;
    Keypoints_transformed.resize(Keypoints.size());

    for (int i = 0; i < Keypoints.size(); i++)
    {
        Keypoints_transformed[i] = Rotation * Keypoints[i] + Translation;
    }

    return Keypoints_transformed;
}

bool klt_ros::estimate2Dtf(const std::vector<cv::KeyPoint> &points1,
                           const std::vector<cv::KeyPoint> &points2,
                           const cv::Mat &descr1,
                           const cv::Mat &descr2,
                           std::vector<cv::DMatch> &good_matches)
{
    std::vector<cv::DMatch> knn_matches;
    knn_simple(points1, points2, descr1, descr2, knn_matches);

    if (knn_matches.size() < MIN_NUM_FEAT)
    {
        ROS_INFO("Not Enough Correspondences to Compute Camera Egomotion");
        return false;
    }

    Eigen::Matrix<double, 3, Eigen::Dynamic> src(3, knn_matches.size());
    Eigen::Matrix<double, 3, Eigen::Dynamic> dst(3, knn_matches.size());

    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        cv::DMatch m = knn_matches[i];
        int qidx = m.queryIdx;
        int tidx = m.trainIdx;

        cv::KeyPoint p1 = points1[qidx];
        cv::KeyPoint p2 = points2[tidx];

        src.col(i) << p1.pt.y, p1.pt.x, 0;
        dst.col(i) << p2.pt.y, p2.pt.x, 0;
    }

    teaserParams2DTFEstimation();
    bool matched=estimateAffineTFTeaser(src, dst, knn_matches, good_matches);

    if(!matched)
        return false;
    
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            R_2D.at<double>(i, j) = R.at<double>(i, j);
        }
    }

    t_2D.at<double>(0) = t.at<double>(0);
    t_2D.at<double>(1) = t.at<double>(1);

    return true;
}

std::vector<cv::KeyPoint> klt_ros::transform2DKeyPoints(const std::vector<cv::KeyPoint> Keypoints, cv::Mat Rotation, cv::Mat Translation)
{
    std::vector<cv::KeyPoint> Keypoints_transformed;
    Keypoints_transformed.resize(Keypoints.size());

    std::vector<cv::Point2f> points, points_transformed;
    points.resize(Keypoints.size());

    cv::Mat tf2d = cv::Mat::eye(3, 3, CV_64F);

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            tf2d.at<double>(i, j) = Rotation.at<double>(i, j);
        }
    }
    tf2d.at<double>(0, 2) = Translation.at<double>(0);
    tf2d.at<double>(1, 2) = Translation.at<double>(1);

    for (int i = 0; i < Keypoints.size(); i++)
    {
        points[i] = Keypoints[i].pt;
    }
    cv::perspectiveTransform(points, points_transformed, tf2d);

    for (int i = 0; i < Keypoints.size(); i++)
    {
        Keypoints_transformed[i].pt = points_transformed[i];
    }

    return Keypoints_transformed;
}

bool klt_ros::estimateAffineTFTeaser(const Eigen::Matrix<double, 3, Eigen::Dynamic> &src,
                                     const Eigen::Matrix<double, 3, Eigen::Dynamic> &dst,
                                     std::vector<cv::DMatch> &initial_matches,
                                     std::vector<cv::DMatch> &good_matches)
{

    solver = new teaser::RobustRegistrationSolver(tparams);
    // Solve with TEASER++
    solver->solve(src, dst);

    auto solution = solver->getSolution();

    if (!solution.valid)
    {
        delete solver;
        std::cout << "Not valid solution" << std::endl;
        return false;
    }

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            R.at<double>(i, j) = solution.rotation(i, j);
            Rot_eig(i, j) = solution.rotation(i, j);
        }
    }
    t.at<double>(0) = solution.translation(0);
    t.at<double>(1) = solution.translation(1);
    t.at<double>(2) = solution.translation(2);
    t_eig(0) = solution.translation(0);
    t_eig(1) = solution.translation(1);
    t_eig(2) = solution.translation(2);

    std::vector<int> inliners = solver->getTranslationInliers();
    float fitness = (float)inliners.size() / (float)initial_matches.size();
    std::cout << "Fitness:" << fitness << std::endl;
    std::cout << "Inliners size:" << inliners.size() << std::endl;
    
    std::cout << "ROT:\n" << Rot_eig<< std::endl;
    std::cout << "TR:\n" << t_eig<< std::endl;

    if(inliners.size()< MIN_NUM_FEAT)
        return false;
    
    for (int i = 0; i < inliners.size(); i++)
    {
        int inidx = inliners[i];
        cv::DMatch m = initial_matches[inidx];
        good_matches.push_back(m);
    }
    delete solver;
    return true;
}

void klt_ros::featureTracking(cv::Mat img_1, cv::Mat img_2, std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2, std::vector<uchar> &status)
{

    //this function automatically gets rid of points for which tracking fails

    std::vector<float> err;
    cv::Size winSize = cv::Size(21, 21);
    cv::TermCriteria termcrit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

    cv::calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    int indexCorrection = 0;
    for (int i = 0; i < status.size(); i++)
    {
        cv::Point2f pt = points2.at(i - indexCorrection);
        if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0))
        {
            if ((pt.x < 0) || (pt.y < 0))
            {
                status.at(i) = 0;
            }
            points1.erase(points1.begin() + (i - indexCorrection));
            points2.erase(points2.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
}

void klt_ros::featureDetection(cv::Mat img_1, std::vector<cv::Point2f> &points1)
{ //uses FAST as of now, modify parameters as necessary

    /*
        std::vector<cv::KeyPoint> keypoints_1;
        int fast_threshold = 20;
        bool nonmaxSuppression = true;
        cv::FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
        cv::KeyPoint::convert(keypoints_1, points1, std::vector<int>());
        */

    int maxCorners = 500;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 6;
    bool useHarrisDetector = false;
    double k = 0.04;
    cv::goodFeaturesToTrack(img_1, points1, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarrisDetector, k);
}

void klt_ros::siftFeatureDetection(const cv::Mat &img_1,
                                   std::vector<cv::KeyPoint> &points1,
                                   cv::Mat &descriptors1)
{
    sift->detectAndCompute(img_1, cv::noArray(), points1, descriptors1);
}

void klt_ros::siftFeatureDetection(const cv::Mat &img_1,
                                   std::vector<cv::KeyPoint> &points1,
                                   cv::Mat &descriptors1,
                                   cv::Mat &depth)
{
    cv::Mat mask = cv::Mat::ones(depth.rows, depth.cols, CV_8U);

    for (int r = 0; r < depth.rows; r++)
    {
        for (int c = 0; c < depth.cols; c++)
        {
            float d = depth.at<float>(r, c);
            if (d < 0.0001f || d != d)
            {
                mask.at<uchar>(r, c) = 0;
            }
        }
    }

    
//     cv::Mat outMasked;
//     cv::bitwise_and(img_1,img_1,outMasked,mask);
//     
//     char buf[32];
//     sprintf(buf, "/tmp/d/mask%d.png", frame);
//     cv::imwrite(buf, outMasked);
    
    sift->detectAndCompute(img_1, mask, points1, descriptors1);
}

void klt_ros::vo()
{
    bool matched = false;
    
    if(!img_inc || !voInitialized)
        return;
    
    if (trackOn)
    {
        trackFeatures();
        cv::Mat mask;
        //Compute Essential Matrix with the Nister Alogirthm
        E = cv::findEssentialMat(currFeatures, prevFeatures, fx, pp, cv::RANSAC, 0.999, 1.0, mask);
        cv::recoverPose(E, currFeatures, prevFeatures, R, t, fx, pp, mask);
        //std::cout << " Rel Vo" << R << " " << t << std::endl;
        
    }
    else
    {
        std::vector<cv::DMatch> good_matches;
        std::vector<cv::KeyPoint> matched_currKeypoints, matched_prevKeypoints, matched_prevKeypoints_transformed;
        cv::Mat matched_prevDescr, matched_currDescr;

        if (!useDepth)
        {
            siftFeatureDetection(currImage, currKeypoints, currDescr);
            matched = estimate2DtfAnd2DPoints(prevKeypoints, currKeypoints,
                                                    prevDescr, currDescr,
                                                    good_matches,
                                                    matched_prevKeypoints, matched_currKeypoints,
                                                    matched_prevKeypoints_transformed);
            if (matched)
            {
                plotTransformedKeypoints(matched_currKeypoints, matched_prevKeypoints_transformed);
                computeTransformedKeypointsError(matched_currKeypoints, matched_prevKeypoints_transformed);
            }
        }
        else
        {
            siftFeatureDetection(currImage, currKeypoints, currDescr, currDepthImage);
            std::vector<Eigen::Vector3d> matched_prevPoints_3D;
            std::vector<Eigen::Vector3d> matched_currPoints_3D;
            std::vector<Eigen::Vector3d> matched_prevPoints_transformed_3D;
            
            if(benchmark_3D)
            {                   
                matched = estimate3Dtf(prevKeypoints, currKeypoints,
                                        prevDescr, currDescr,
                                        good_matches,
                                        matched_prevKeypoints, matched_currKeypoints,
                                        matched_prevPoints_3D, matched_currPoints_3D,
                                        matched_prevPoints_transformed_3D);
                if(!matched)
                    return;

                Eigen::Affine3d delta;
                delta.translation() = t_eig;
                delta.linear() = Rot_eig;
                delta=delta.inverse();
                curr_pose = delta*curr_pose;
                
                addTfToPath(curr_pose);                
                
            }
            else
            {
                matched=estimate2DtfAnd3DPoints(prevKeypoints, currKeypoints,
                                    prevDescr, currDescr,
                                    good_matches,
                                    matched_prevKeypoints, matched_currKeypoints,
                                    matched_prevKeypoints_transformed,
                                    matched_prevPoints_3D, matched_currPoints_3D,
                                    matched_prevPoints_transformed_3D);
                
                if(!matched)
                    return;
                
                computeTransformedKeypoints3DError(matched_currPoints_3D, matched_prevPoints_transformed_3D);
            }


        }

        if(publish_matches)
            publishMatches(prevImage, currImage, prevKeypoints, currKeypoints, good_matches);
        
        show_matches(prevImage, currImage, prevKeypoints, currKeypoints, good_matches);
    }
    
    double scale = 1.0;
    if ((t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1)))
    {
        t_f = t_f + scale * R_f * t;
        R_f = R * R_f;
    }
    

    prevImage = currImage.clone();
    prevDepthImage = currDepthImage.clone();

    std::swap(prevKeypoints, currKeypoints);
    std::swap(prevDescr, currDescr);

    std::cout << "VO" << std::endl;
    std::cout << t_f << std::endl;
    std::cout << R_f << std::endl;
    img_inc = false;        
}

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
/* Helper Functions */
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
void klt_ros::addTfToPath(const Eigen::Affine3d &vision_pose)
{
    Eigen::Affine3d pose=fromVisionCord(vision_pose);
    Eigen::Quaterniond quat(pose.linear());

    geometry_msgs::PoseStamped ps;
    ps.header.stamp = ros::Time::now();
    ps.header.frame_id = "odom";
    ps.pose.position.x=pose.translation()(0);
    ps.pose.position.y=pose.translation()(1);
    ps.pose.position.z=pose.translation()(2);
    
    ps.pose.orientation.x=quat.x();
    ps.pose.orientation.y=quat.y();
    ps.pose.orientation.z=quat.z();
    ps.pose.orientation.w=quat.w();
    
    odomPath.poses.push_back(ps);
}

void klt_ros::publishOdomPath()
{    
    nav_msgs::Path newPath=odomPath;
    newPath.header.stamp = ros::Time::now();
    newPath.header.frame_id = "odom";
    odom_path_pub.publish(newPath);
}

sensor_msgs::Image image;

void klt_ros::publishMatches(const cv::Mat &img_1,
                           const cv::Mat &img_2,
                           const std::vector<cv::KeyPoint> keypoints1,
                           const std::vector<cv::KeyPoint> keypoints2,
                           const std::vector<cv::DMatch> &good_matches)
{
    static int counter=0;
    
    if (good_matches.size() == 0)
        return;

    cv::Mat img_matches;
    cv::drawMatches(img_1, keypoints1,
                    img_2, keypoints2,
                    good_matches,
                    img_matches,
                    cv::Scalar::all(-1),
                    cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::DEFAULT);
    
    sensor_msgs::Image ros_msg;
    
    std_msgs::Header header; // empty header
    header.seq = counter; // user defined counter
    header.stamp = ros::Time::now(); // time
    
    cv_bridge::CvImage img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, img_matches);
    img_bridge.toImageMsg(ros_msg); 
    matches_pub.publish(ros_msg);
}

void klt_ros::show_matches(const cv::Mat &img_1,
                           const cv::Mat &img_2,
                           const std::vector<cv::KeyPoint> keypoints1,
                           const std::vector<cv::KeyPoint> keypoints2,
                           const std::vector<cv::DMatch> &good_matches)
{
    if (good_matches.size() == 0)
        return;

    cv::Mat img_matches;
    cv::drawMatches(img_1, keypoints1,
                    img_2, keypoints2,
                    good_matches,
                    img_matches,
                    cv::Scalar::all(-1),
                    cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::DEFAULT);

    std::string filename = output_path+ "/matches/good_matches_" + std::to_string(frame)+".png";

    cv::imwrite(filename, img_matches);
}

void klt_ros::plotTransformedKeypoints(std::vector<cv::KeyPoint> matched_currKeypoints, std::vector<cv::KeyPoint> matched_prevKeypoints_transformed)
{

    if(matched_currKeypoints.size() == 0)
        return;
    for (size_t i = 0; i < matched_prevKeypoints_transformed.size(); i++)
    {
        cv::circle(currImageRGB, matched_prevKeypoints_transformed[i].pt, 5, CV_RGB(255, 0, 0), 1);
        cv::circle(currImageRGB, matched_currKeypoints[i].pt, 5, CV_RGB(0, 255, 0), -1);
    }

    std::string filename = output_path+ "/keypoints/transformed_keypoints_" + std::to_string(frame)+".png";

    cv::imwrite(filename, currImageRGB);
}

void klt_ros::computeTransformedKeypoints3DError(std::vector<Eigen::Vector3d> matched_currKeypoints_3D, std::vector<Eigen::Vector3d> matched_prevKeypoints_transformed_3D)
{
    if(matched_currKeypoints_3D.size() == 0)
        return;

    Eigen::VectorXd errorX, errorY, errorZ;
    std::cout << matched_prevKeypoints_transformed_3D.size() << " " << matched_currKeypoints_3D.size() << std::endl;
    errorX.resize(matched_prevKeypoints_transformed_3D.size());
    errorY.resize(matched_prevKeypoints_transformed_3D.size());
    errorZ.resize(matched_prevKeypoints_transformed_3D.size());
    for (size_t i = 0; i < matched_prevKeypoints_transformed_3D.size(); i++)
    {
        errorX(i) = fabs(matched_currKeypoints_3D[i](0) - matched_prevKeypoints_transformed_3D[i](0));
        errorY(i) = fabs(matched_currKeypoints_3D[i](1) - matched_prevKeypoints_transformed_3D[i](1));
        errorZ(i) = fabs(matched_currKeypoints_3D[i](2) - matched_prevKeypoints_transformed_3D[i](2));
    }

    std::string filename = output_path+ "/keypoint3D_error/error3D_" + std::to_string(frame)+".txt";
    std::ofstream file(filename);
    if (file.is_open())
    {
        file << "ErrorX=[" << errorX << "]"<< std::endl;
        file << "ErrorY=[" << errorY << "]"<< std::endl;
        file << "ErrorZ=[" << errorZ << "]"<< std::endl;

        file << "MeanErrorX=" << errorX.mean() << std::endl;
        file << "MeanErrorY=" << errorY.mean() << std::endl;
        file << "MeanErrorZ=" << errorZ.mean() << std::endl;

        file << "MaxErrorX=" << errorX.maxCoeff() << std::endl;
        file << "MaxErrorY=" << errorY.maxCoeff() << std::endl;
        file << "MaxErrorZ=" << errorZ.maxCoeff() << std::endl;

        file << "MinErrorX=" << errorX.minCoeff() << std::endl;
        file << "MinErrorY=" << errorY.minCoeff() << std::endl;
        file << "MinErrorZ=" << errorZ.minCoeff() << std::endl;

    }
}

void klt_ros::computeTransformedKeypointsError(std::vector<cv::KeyPoint> matched_currKeypoints,
                                               std::vector<cv::KeyPoint> matched_prevKeypoints_transformed)
{
    if(matched_currKeypoints.size() == 0)
        return;
    Eigen::VectorXd errorX, errorY;
    errorX.resize(matched_prevKeypoints_transformed.size());
    errorY.resize(matched_prevKeypoints_transformed.size());

    for (size_t i = 0; i < matched_prevKeypoints_transformed.size(); i++)
    {
        errorX(i) = fabs(matched_currKeypoints[i].pt.y - matched_prevKeypoints_transformed[i].pt.y);
        errorY(i) = fabs(matched_currKeypoints[i].pt.x - matched_prevKeypoints_transformed[i].pt.x);
    }

    std::string filename = output_path+ "/keypoint_error/error_" + std::to_string(frame)+".txt";

    std::ofstream file(filename);
    if (file.is_open())
    {
        file << "PixelErrorX=[" << errorX << "]"<< std::endl;
        file << "PixelErrorY=[" << errorY << "]"<< std::endl;

        file << "MeanPixelErrorX=" << errorX.mean() << std::endl;
        file << "MeanPixelErrorY=" << errorY.mean() << std::endl;

        file << "MaxPixelErrorX=" << errorX.maxCoeff() << std::endl;
        file << "MaxPixelErrorY=" << errorY.maxCoeff() << std::endl;

        file << "MinPixelErrorX=" << errorX.minCoeff() << std::endl;
        file << "MinPixelErrorY=" << errorY.minCoeff() << std::endl;
    }
}

void klt_ros::plotFeatures()
{

    for (size_t i = 0; i < prevFeatures.size(); i++)
    {
        cv::circle(prevImage, prevFeatures[i], 10, cv::Scalar(255.), -1);
    }

    cv::namedWindow("Good Featurse to Track", CV_WINDOW_AUTOSIZE);
    cv::imshow("Good Features to Track", prevImage);
    cv::waitKey(0);
}

void klt_ros::knn_mutual(std::vector<cv::KeyPoint> keypoints1,
                         std::vector<cv::KeyPoint> keypoints2,
                         cv::Mat des1,
                         cv::Mat des2,
                         std::vector<cv::DMatch> &good_matches)
{
    good_matches.clear();
    cv::Ptr<cv::FlannBasedMatcher> matcher = cv::FlannBasedMatcher::create();

    std::vector<std::vector<cv::DMatch>> knn_matches1, knn_matches2;
    matcher->knnMatch(des1, des2, knn_matches1, 1);
    matcher->knnMatch(des2, des1, knn_matches2, 1);

    std::sort(knn_matches2.begin(), knn_matches2.end(), orderVec);
    for (size_t i = 0; i < knn_matches1.size(); i++)
    {
        int tidx1 = knn_matches1[i][0].trainIdx;
        int qidx1 = knn_matches1[i][0].queryIdx;

        int qidx2 = knn_matches2[tidx1][0].queryIdx;
        int tidx2 = knn_matches2[tidx1][0].trainIdx;

        std::cout << tidx1 << " " << qidx1 << " " << qidx2 << " " << tidx2 << std::endl;
        if (qidx1 == tidx2)
        {

            good_matches.push_back(knn_matches1[i][0]);
        }
    }
}

void klt_ros::knn(std::vector<cv::KeyPoint> keypoints1,
                  std::vector<cv::KeyPoint> keypoints2,
                  cv::Mat des1,
                  cv::Mat des2,
                  std::vector<cv::DMatch> &good_matches)
{
    cv::Ptr<cv::FlannBasedMatcher> matcher = cv::FlannBasedMatcher::create();

    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(des1, des2, knn_matches, 2);

    const float ratio_thresh = 0.7f;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        double dist = knn_matches[i][0].distance;
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
}

void klt_ros::knn_simple(std::vector<cv::KeyPoint> keypoints1,
                         std::vector<cv::KeyPoint> keypoints2,
                         cv::Mat des1,
                         cv::Mat des2,
                         std::vector<cv::DMatch> &good_matches)
{
    good_matches.clear();
    cv::Ptr<cv::FlannBasedMatcher> matcher = cv::FlannBasedMatcher::create();

    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(des1, des2, knn_matches, 1);

    //const float max_dist = 100;
    const float max_dist = 80;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < max_dist)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
}

void klt_ros::teaserParams2DTFEstimation()
{
    tparams.noise_bound = 10;
    tparams.cbar2 = 1;
    tparams.estimate_scaling = false;
    tparams.rotation_max_iterations = 100;
    tparams.rotation_gnc_factor = 1.4;
    tparams.rotation_estimation_algorithm = teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    tparams.rotation_cost_threshold = 0.005;
}

void klt_ros::teaserParams3DTFEstimation()
{
//     tparams.noise_bound = 0.01;
//     tparams.cbar2 = 1;
//     tparams.estimate_scaling = false;
//     tparams.rotation_max_iterations = 100;
//     tparams.rotation_gnc_factor = 1.4;
//     tparams.rotation_estimation_algorithm = teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
//     tparams.rotation_cost_threshold = 0.005;
            
    tparams.noise_bound =  0.05;
    tparams.cbar2 = 1;
    tparams.estimate_scaling = false;
    tparams.rotation_max_iterations = 100;
    tparams.rotation_gnc_factor = 1.4;
    tparams.rotation_estimation_algorithm =
    teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    tparams.rotation_cost_threshold = 0.005;

    
}

void klt_ros::trackFeatures()
{

    //a redetection is triggered in case the number of feautres being trakced go below a particular threshold
    if (prevFeatures.size() < MIN_NUM_FEAT)
    {
        //cout << "Number of tracked features reduced to " << prevFeatures.size() << endl;
        //cout << "trigerring redection" << endl;
        featureDetection(prevImage, prevFeatures);
    }
    featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
}
