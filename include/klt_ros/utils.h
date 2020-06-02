#ifndef UTILS_H
#define UTILS_H

#include <eigen3/Eigen/Dense>

Eigen::Matrix4d T_B_P;

inline Eigen::Affine3d fromVisionCord(const Eigen::Affine3d &pose)
{
    T_B_P << 0,-1,  0 ,0, 
             0, 0, -1, 0, 
             1, 0,  0, 0, 
             0, 0,  0, 1;
             
    Eigen::Matrix4d invT_B_P=T_B_P.inverse();
    
    Eigen::Affine3d ret;
    ret.matrix() = invT_B_P*pose*T_B_P;
    return ret;
} 

bool orderVec(const std::vector<cv::DMatch> &v1, const std::vector<cv::DMatch> &v2)
{
    return v1[0].queryIdx < v2[0].queryIdx;
}

#endif