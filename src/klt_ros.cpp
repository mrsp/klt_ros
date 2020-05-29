#include <klt_ros/klt_ros.h>

klt_ros::klt_ros(ros::NodeHandle nh_) : it_(nh_)
{
 


    sift = cv::xfeatures2d::SIFT::create();

    img_inc = false;
    firstImageCb = true;
    firstCameraInfoCb = true;
    MIN_NUM_FEAT = 200;

    trackOn = false;
    voInitialized = false;

    frame = 0;
    R_f = cv::Mat::eye(3, 3, CV_64F);
    t_f = cv::Mat::zeros(3, 1, CV_64F);
    R_2D = cv::Mat::eye(3, 3, CV_64F);
    t_2D = cv::Mat::zeros(3, 1, CV_64F);
    R = cv::Mat::eye(3, 3, CV_64F);
    t = cv::Mat::zeros(3, 1, CV_64F);
    
    ros::NodeHandle n_p("~");

    n_p.params<std::string>("image_topic",image_topic,"/camera/rgb/image_rect_color");
    n_p.params<bool>("useDepth",useDepth,false);
    n_p.params<std::string>("depth_topic",depth_topic,"/camera/depth_registered/image_rect");
    n_p.params<std::string>("cam_info_topic",cam_info_topic, "/camera/rgb/camera_info");

    if(useDepth)
    {        
        image_sub = new message_filters::Subscriber<sensor_msgs::Image>(it_,image_topic,1);
        depth_sub = new message_filters::Subscriber<sensor_msgs::Image>(it_,depth_topic,1);

        ts_sync = new Synchronizer<MySyncPolicy>(MySyncPolicy(10),image_sub,depht_sub);

        ts_sync->registerCallback(boost::bind(&klt_ros::imageDepthCb,this,_1,_2));
    }
    else
        image_sub_ = it_.subscribe(image_topic, 1,  &klt_ros::imageCb, this);

    ROS_INFO("Waiting camera info");
    while(ros::ok())
    {
        sensor_msgs::CameraInfoConstPtr cam_info=ros::topic::waitForMessage<sensor_msgs::CameraInfo>(cam_info_topic);
        if(cam_info)
        {
            cameraInfoCb(cam_info);
            break;
        }
    }
}


void klt_ros::imageDepthCb(const sensor_msgs::ImageConstPtr &img_msg,const sensor_msgs::CameraInfoConstPtr &depth_msg)
{
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



    if (firstImageCb)
    {
        //prevImage = cv_ptr->image;
        cvtColor(cv_ptr->image, prevImage, cv::COLOR_BGR2GRAY);
        prevDepthImage = cv_depth_ptr->image;
        siftFeatureDetection(prevImage, prevKeypoints, prevDescr);

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

    if(firstCameraInfoCb)
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
    tparams.noise_bound = 0.01;
    tparams.cbar2 = 1;
    tparams.estimate_scaling = false;
    tparams.rotation_max_iterations = 100;
    tparams.rotation_gnc_factor = 1.4;
    tparams.rotation_estimation_algorithm = teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    tparams.rotation_cost_threshold = 0.005;
}

bool klt_ros::estimate2Dtf(const std::vector<cv::KeyPoint> &points1,
                          const std::vector<cv::KeyPoint> &points2,
                          const cv::Mat &prevDescr,
                          const cv::Mat &currDescr,
                          std::vector<cv::DMatch> &good_matches, 
                          std::vector<cv::KeyPoint>& m_points1, 
                           std::vector<cv::KeyPoint>& m_points2, 
                           std::vector<cv::KeyPoint> &m_points1_transformed, 
                           cv::Mat& m_d1, 
                           cv::Mat& m_d2)
{
    std::vector<cv::DMatch> knn_matches;
    knn_simple(points1, points2, prevDescr, currDescr, knn_matches);

    if (knn_matches.size() < 10)
    {
        ROS_INFO("Not Enough Correspondences to Compute Camera Egomotion");
        return false;
    }

    Eigen::Matrix<double, 3, Eigen::Dynamic> src(3, knn_matches.size());
    Eigen::Matrix<double, 3, Eigen::Dynamic> dst(3, knn_matches.size());

    m_points1.resize(knn_matches.size());
    m_points2.resize(knn_matches.size());
    m_d1 = cv::Mat::zeros(knn_matches.size(),prevDescr.cols,CV_64F);
    m_d2 = cv::Mat::zeros(knn_matches.size(),prevDescr.cols,CV_64F);

    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        cv::DMatch m = knn_matches[i];
        int qidx = m.queryIdx;
        int tidx = m.trainIdx;

        cv::KeyPoint p1 = points1[qidx];
        cv::KeyPoint p2 = points2[tidx];


        src.col(i) << p1.pt.x, p1.pt.y, 0;
        dst.col(i) << p2.pt.x, p2.pt.y, 0;


    }


    teaserParams2DTFEstimation();
    estimateAffineTFTeaser(src, dst, knn_matches, good_matches);

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            R_2D.at<double>(i, j) = R.at<double>(i, j);
        }
    }
    
    for (size_t i = 0; i < good_matches.size(); i++)
    {
        cv::DMatch m=good_matches[i];
        int qidx = m.queryIdx;
        int tidx = m.trainIdx;

        cv::KeyPoint p1 = points1[qidx];
        cv::KeyPoint p2 = points2[tidx];
        
        cv::Mat d1 = prevDescr.row(qidx);
        cv::Mat d2 = currDescr.row(tidx);

        m_points1[i] = p1;
        m_points2[i] = p2;
        m_d1.row(i) = d1;
        m_d2.row(i) = d1;
    }
    
    
    t_2D.at<double>(0) = t.at<double>(0);
    t_2D.at<double>(1) = t.at<double>(1);
    
    m_points1_transformed = transform2DKeyPoints(m_points1, R_2D, t_2D);

    return true; 
}

std::vector<cv::KeyPoint> klt_ros::transform2DKeyPoints(const std::vector<cv::KeyPoint> Keypoints, cv::Mat Rotation, cv::Mat Translation)
{
    std::vector<cv::KeyPoint> Keypoints_transformed;
    Keypoints_transformed.resize(Keypoints.size());

    std::vector<cv::Point2f> points, points_transformed;
    points.resize(Keypoints.size());
    
    cv::Mat tf2d = cv::Mat::eye(3,3,CV_64F);

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            tf2d.at<double>(i,j) = Rotation.at<double>(i,j);
        }
    }
    tf2d.at<double>(0,2) = Translation.at<double>(0);
    tf2d.at<double>(1,2) = Translation.at<double>(1);



    for (int i = 0; i < Keypoints.size(); i++)
    {
        points[i] = Keypoints[i].pt;
    }
    cv::perspectiveTransform(points,points_transformed,tf2d);


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
        }
    }
    t.at<double>(0) = solution.translation(0);
    t.at<double>(1) = solution.translation(1);
    t.at<double>(2) = solution.translation(2);

    std::cout << "Teaser " << R_f << " " << t_f << std::endl;

    std::vector<int> inliners = solver->getTranslationInliers();
    float fitness = (float)inliners.size() / (float)initial_matches.size();
    std::cout << "Fitness:" << fitness << std::endl;
    std::cout << "Inliners size:" << inliners.size() << std::endl;

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

    //     cv::imshow("Good Matches", img_matches );
    //     cv::waitKey(0);

    char buf[32];
    sprintf(buf, "/tmp/d/good_matches_%d.png", frame);
    cv::imwrite(buf, img_matches);
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

/*
void klt_ros::estimate2Dtf(std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2)
{
    int N = min(points1.size(), points2.size());
    Eigen::Matrix<double, 3, Eigen::Dynamic> src(3, N);
    Eigen::Matrix<double, 3, Eigen::Dynamic> dst(3, N);

    for (size_t i = 0; i < N; ++i)
    {
        src.col(i) << points1[i].x, points1[i].y, 0.0;
        dst.col(i) << points2[i].x, points2[i].y, 0.0;
    }

    estimateAffineTFTeaser(src, dst);
}
*/

void klt_ros::vo()
{

    if (img_inc && voInitialized)
    {
        if (trackOn)
        {
            trackFeatures();
            cv::Mat mask;
            E = cv::findEssentialMat(currFeatures, prevFeatures, fx, pp, cv::RANSAC, 0.999, 1.0, mask);

            cv::recoverPose(E, currFeatures, prevFeatures, R, t, fx, pp, mask);
            //std::cout << " Rel Vo" << R << " " << t << std::endl;
        }
        else
        {
            //featureDetection(currImage, currFeatures);
            siftFeatureDetection(currImage, currKeypoints, currDescr);
            
            std::vector<cv::DMatch> good_matches;
            std::vector<cv::KeyPoint>  matched_currKeypoints, matched_prevKeypoints, matched_prevKeypoints_transformed;
            cv::Mat matched_prevDescr, matched_currDescr;

            bool matched=estimate2Dtf(prevKeypoints, currKeypoints,
                                      prevDescr, currDescr, 
                                      good_matches, 
                                      matched_prevKeypoints, matched_currKeypoints, 
                                      matched_prevKeypoints_transformed, 
                                      matched_prevDescr, matched_currDescr);

            if(matched)
            {
                plotTransformedKeypoints(matched_currKeypoints, matched_prevKeypoints_transformed);
                computeTransformedKeypointsError(matched_currKeypoints, matched_prevKeypoints_transformed);
            }

            //show_matches(prevImage, currImage, prevKeypoints, currKeypoints, good_matches);

        }

        if ((t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1)))
        {
            t_f = t_f + R_f * t;
            R_f = R * R_f;
        }

        prevImage = currImage.clone();
        std::swap(prevKeypoints, currKeypoints);
        std::swap(prevDescr, currDescr);

        std::cout << "VO" << std::endl;
        std::cout << t_f << std::endl;
        std::cout << R_f << std::endl;
        img_inc = false;
    }
}
void klt_ros::plotTransformedKeypoints(std::vector<cv::KeyPoint> matched_currKeypoints, std::vector<cv::KeyPoint> matched_prevKeypoints_transformed)
{

    for (size_t i = 0; i < matched_prevKeypoints_transformed.size(); i++)
    {
        cv::circle(currImageRGB, matched_prevKeypoints_transformed[i].pt, 5, CV_RGB(255,0,0), 1);
        cv::circle(currImageRGB, matched_currKeypoints[i].pt, 5, CV_RGB(0,255,0), -1);
    }

  
     char buf[128];
    sprintf(buf, "/tmp/keypoints/transformed_keypoints_%d.png", frame);
    cv::imwrite(buf, currImageRGB);
}

void klt_ros::computeTransformedKeypointsError(std::vector<cv::KeyPoint> matched_currKeypoints, std::vector<cv::KeyPoint> matched_prevKeypoints_transformed)
{ 
    Eigen::VectorXd errorX, errorY;
    errorX.resize(matched_prevKeypoints_transformed.size());
    errorY.resize(matched_prevKeypoints_transformed.size());

    for (size_t i = 0; i < matched_prevKeypoints_transformed.size(); i++)
    {
        errorX(i) = abs(matched_currKeypoints[i].pt.x - matched_prevKeypoints_transformed[i].pt.x);
        errorY(i) = abs(matched_currKeypoints[i].pt.y - matched_prevKeypoints_transformed[i].pt.y);
    }

    
  
    char buf[128];
    sprintf(buf, "/tmp/keypoint_error/error_%d.txt", frame);
    std::ofstream file(buf);
    if (file.is_open())
    {
        file << "Error (Pixels)" << errorX << "\t" << errorY<< std::endl;
        file << "Mean Error (Pixels) "<< errorX.mean() << "\t "<< errorY.mean()<<std::endl;
        file << "Max Error (Pixels) "<< errorX.maxCoeff() << "\t "<< errorY.maxCoeff()<<std::endl;
        file << "Min Error (Pixels) "<< errorX.minCoeff() << "\t "<< errorY.minCoeff()<<std::endl;
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

bool orderVec(const std::vector<cv::DMatch> &v1, const std::vector<cv::DMatch> &v2)
{
    return v1[0].queryIdx < v2[0].queryIdx;
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
