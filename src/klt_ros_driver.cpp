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
    static ros::Rate rate(100);
    while (ros::ok())
    {
        klt.vo();
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}