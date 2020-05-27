#include <klt_ros/klt_ros.h>
#include <opencv2/xfeatures2d/nonfree.hpp>

klt_ros::klt_ros(ros::NodeHandle nh_) : it_(nh_)
{
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/camera/rgb/image_rect_color", 1,
                               &klt_ros::imageCb, this);
    
    img_inc = false;
    firstImageCb = true;
    MIN_NUM_FEAT = 200;
    focal = 570.3422241210938;
    pp.x = 319.5;
    pp.y = 239.5;
    trackOn = false;
    voInitialized = false;
    
    sift = cv::xfeatures2d::SIFT::create();
    initTeaserParams();
    solver = new teaser::RobustRegistrationSolver(tparams);
}

void klt_ros::knn(std::vector<cv::KeyPoint> keypoints1,
         std::vector<cv::KeyPoint> keypoints2,
         cv::Mat des1,
         cv::Mat des2,
         std::vector<cv::DMatch> &good_matches)
{
    cv::Ptr<cv::FlannBasedMatcher> matcher = cv::FlannBasedMatcher::create();

    std::vector< std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch( des1, des2, knn_matches, 2 );
    
    const float ratio_thresh = 0.7f;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        double dist=knn_matches[i][0].distance;
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

    std::vector< std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch( des1, des2, knn_matches, 1 );
    
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

bool orderVec (const std::vector<cv::DMatch> &v1,const std::vector<cv::DMatch> &v2) 
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

    std::vector< std::vector<cv::DMatch> > knn_matches1,knn_matches2;
    matcher->knnMatch( des1, des2, knn_matches1, 1 );
    matcher->knnMatch( des2, des1, knn_matches2, 1 );
    
    std::sort (knn_matches2.begin(), knn_matches2.end(),orderVec);
    for (size_t i = 0; i < knn_matches1.size(); i++)
    {
        int tidx1=knn_matches1[i][0].trainIdx;
        int qidx1=knn_matches1[i][0].queryIdx;
        
        int qidx2=knn_matches2[tidx1][0].queryIdx;
        int tidx2=knn_matches2[tidx1][0].trainIdx;
        
        
        std::cout<<tidx1<<" "<<qidx1<<" "<<qidx2<<" "<<tidx2<<std::endl;
        if (qidx1==tidx2)
        {
            
            good_matches.push_back(knn_matches1[i][0]);            
        }
    }
}

void klt_ros::initTeaserParams()
{
    tparams.noise_bound = 10;
    tparams.cbar2 = 1;
    tparams.estimate_scaling = false;
    tparams.rotation_max_iterations = 100;
    tparams.rotation_gnc_factor = 1.4;
    tparams.rotation_estimation_algorithm = teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    tparams.rotation_cost_threshold = 0.005;
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

    if(!solution.valid)
    {
        std::cout<<"Not valid solution"<<std::endl;
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
    
    std::vector<int> inliners=solver->getTranslationInliers();
    float fitness=(float)inliners.size()/(float)initial_matches.size();
    std::cout << "Fitness:"<<fitness << std::endl;
    std::cout << "Inliners size:"<< inliners.size()<<std::endl;

    for(int i=0;i<inliners.size();i++)
    {
        int inidx=inliners[i];
        cv::DMatch m=initial_matches[inidx];
        good_matches.push_back(m);
    }
    
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
    sift->detectAndCompute(img_1,cv::noArray(), points1,descriptors1);  
}

void klt_ros::show_matches(const cv::Mat &img_1,
                           const cv::Mat &img_2,
                           const std::vector<cv::KeyPoint> keypoints1,
                           const std::vector<cv::KeyPoint> keypoints2,
                           const std::vector<cv::DMatch> &good_matches)
{
    cv::Mat img_matches;
    cv::drawMatches(img_1,keypoints1, 
                    img_2,keypoints2,
                    good_matches,
                    img_matches, 
                    cv::Scalar::all(-1),
                    cv::Scalar::all(-1), 
                    std::vector<char>(), 
                    cv::DrawMatchesFlags::DEFAULT );
    
    cv::imshow("Good Matches", img_matches );
    cv::waitKey(0);
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
*/

bool klt_ros::compute2Dtf(const std::vector<cv::KeyPoint> &points1,
                          const std::vector<cv::KeyPoint> &points2,
                          const cv::Mat &prevDescr,
                          const cv::Mat &currDescr,
                          std::vector<cv::DMatch> &good_matches)
{
    std::vector<cv::DMatch> knn_matches;
    knn_simple(points1,points2,prevDescr,currDescr,knn_matches);    


    Eigen::Matrix<double, 3, Eigen::Dynamic>src(3,knn_matches.size());
    Eigen::Matrix<double, 3, Eigen::Dynamic>dst(3,knn_matches.size());

    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        cv::DMatch m=knn_matches[i];
        int qidx=m.queryIdx;
        int tidx=m.trainIdx;

        cv::KeyPoint p1=points1[tidx];
        cv::KeyPoint p2=points2[qidx];
        
        Eigen::Vector3d v1(p1.pt.x,p1.pt.y,0);
        Eigen::Vector3d v2(p2.pt.x,p2.pt.y,0);        
        
        src.col(i)<<v1;
        dst.col(i)<<v2;
        
        std::cout<<"Q:"<<knn_matches[i].queryIdx<<" T:"<<knn_matches[i].trainIdx<<std::endl;
        std::cout<<"\t("<<v1(0)<<','<<v1(1)<<','<<v1(2);
        std::cout<<"\t("<<v2(0)<<','<<v2(1)<<','<<v2(2)<<std::endl;
        
        
    }

    std::cout<<"SRC:"<<src.size()<<std::endl;
    std::cout<<"DST:"<<dst.size()<<std::endl;
    
    return estimateAffineTFTeaser(src, dst,knn_matches,good_matches);
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
        siftFeatureDetection(prevImage, prevKeypoints,prevDescr);
        
        firstImageCb = false;
        R_f = cv::Mat::eye(3, 3, CV_64F);
        t_f = cv::Mat::zeros(3, 1, CV_64F);
        R = cv::Mat::eye(3, 3, CV_64F);
        t = cv::Mat::zeros(3, 1, CV_64F);
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

    if (img_inc && voInitialized)
    {
        if (trackOn)
        {
            trackFeatures();
            cv::Mat mask;
            E = cv::findEssentialMat(currFeatures, prevFeatures, focal, pp, cv::RANSAC, 0.999, 1.0, mask);

            cv::recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);
            //std::cout << " Rel Vo" << R << " " << t << std::endl;
        }
        else
        {
            //featureDetection(currImage, currFeatures);
            siftFeatureDetection(currImage, currKeypoints,currDescr);
            std::vector<cv::DMatch> good_matches;
            compute2Dtf(prevKeypoints,currKeypoints,prevDescr,currDescr,good_matches);
            
            //show_matches(prevImage,currImage,prevKeypoints,currKeypoints,good_matches);
        }

  

        if ((t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1)))
        {
            t_f = t_f + R_f * t;
            R_f = R * R_f;
        }

        prevImage = currImage.clone();
        std::swap(prevKeypoints,currKeypoints);
        std::swap(prevDescr,currDescr);

        std::cout << "VO" << std::endl;
        std::cout << t_f << std::endl;
        std::cout << R_f << std::endl;
        img_inc = false;
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