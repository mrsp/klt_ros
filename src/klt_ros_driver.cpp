/*
 * main.cpp.
 *
 * Written by: Stylianos Piperakis.
 *
 * This file creates an over-segmentation of a provided image based on the SLIC
 * superpixel algorithm, as implemented in slic.h and slic.cpp by Pascal Metter.
 */
#include <klt_ros/klt_ros.h>


int main(int argc, char *argv[])
{

    ros::init(argc, argv, "klt_ros");
    ros::NodeHandle n;
    klt_ros klt(n);
    ros::NodeHandle n_p("~");
    double image_freq;
    n_p.param<double>("image_freq", image_freq, 100.0);
    static ros::Rate rate(2.0*image_freq);
    while (ros::ok())
    {
        klt.vo();
        
        if(klt.isUsingDepth())
        {
            klt.publishOdomPath();
        }
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}