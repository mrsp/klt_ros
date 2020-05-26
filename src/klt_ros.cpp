#include <klt_ros/klt_ros.h>

klt_ros::klt_ros() : it_(nh_)
{
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/camera/rgb/image_rect_color", 1,
                               &klt_ros::imageCb, this);

    firstImageCb = true;
    MIN_NUM_FEAT = 200;
    focal = 570.3422241210938;
    pp.x = 319.5;
    pp.y = 239.5;
    trackOn = false;
}

void klt_ros::initTeaserParams()
{
    tparams.noise_bound = 0.05;
    tparams.cbar2 = 1;
    tparams.estimate_scaling = false;
    tparams.rotation_max_iterations = 100;
    tparams.rotation_gnc_factor = 1.4;
    tparams.rotation_estimation_algorithm = teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    tparams.rotation_cost_threshold = 0.005;
}

void klt_ros::estimateAffineTFTeaser(Eigen::Matrix<double, 3, Eigen::Dynamic> src, Eigen::Matrix<double, 3, Eigen::Dynamic> dst)
{
    // Solve with TEASER++
    solver = new teaser::RobustRegistrationSolver(tparams);

    solver->solve(src, dst);
    auto solution = solver->getSolution();

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            R_f.at<double>(i, j) = solution.rotation(i, j);
        }
    }
    t_f.at<double>(0) = solution.translation(0);
    t_f.at<double>(1) = solution.translation(1);
    t_f.at<double>(2) = solution.translation(2);

    std::cout << "Teaser " << R_f << " " << t_f << std::endl;
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

void klt_ros::compute2Dtf(std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2)
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
void klt_ros::imageCb(const sensor_msgs::ImageConstPtr &msg)
{
    cv_bridge::CvImagePtr cv_ptr;

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
        featureDetection(prevImage, prevFeatures);
        firstImageCb = false;
        R_f = cv::Mat::eye(3, 3, CV_64F);
        t_f = cv::Mat::zeros(3, 1, CV_64F);
    }
    else
    {
        //currImage = cv_ptr->image;
        cvtColor(cv_ptr->image, currImage, cv::COLOR_BGR2GRAY);
        if (!voInitialized)
            voInitialized = true;
    }
}

void klt_ros::vo()
{
    if (voInitialized)
    {
        if (trackOn)
        {
            trackFeatures();
        }
        else
        {
            featureDetection(currImage, currFeatures);
            compute2Dtf(currFeatures, prevFeatures);
        }

        //std::cout << " Feat Size " << std::endl;
        //std::cout << currFeatures.size() << std::endl;

        //recovering the pose and the essential matrix
        cv::Mat mask;
        E = cv::findEssentialMat(currFeatures, prevFeatures, focal, pp, cv::RANSAC, 0.999, 1.0, mask);

        cv::recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);
        //std::cout << " Rel Vo" << R << " " << t << std::endl;

        if ((t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1)))
        {
            t_f = t_f + R_f * t;
            R_f = R * R_f;
        }

        prevImage = currImage.clone();
        prevFeatures = currFeatures;

        std::cout << "VO" << std::endl;
        std::cout << t_f << std::endl;
        std::cout << R_f << std::endl;
    }
}
void klt_ros::plotFeatures()
{

    for (size_t i = 0; i < prevFeatures.size(); i++)
    {
        cv::circle(prevImage, prevFeatures[i], 10, cv::Scalar(255.), -1);
    }

    cv::namedWindow("Good Features to Track", CV_WINDOW_AUTOSIZE);
    cv::imshow("Good Features to Track", prevImage);
    cv::waitKey(0);
}