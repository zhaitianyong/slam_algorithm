//
// Created by atway on 2020/12/21.
//

#ifndef SLAM_ALGORITHM_STEREO_H
#define SLAM_ALGORITHM_STEREO_H
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>


namespace SLAM_STEREO
{

// 向量的叉乘
Eigen::Matrix3d corss(Eigen::Vector3d& t);

// 提取棋盘格角点
bool findCorners(cv::Mat& img, cv::Size& patternSize, std::vector<cv::Point2f> &corners,cv::Mat& drawMat);

// 三角化
void triangulatePoint(Eigen::Matrix<double, 3, 4>& T1, Eigen::Matrix<double, 3, 4>& T2, Eigen::Vector3d& p1, Eigen::Vector3d&p2, Eigen::Vector3d& out);

// 4中姿态中，选择两个相机之前的。
bool choose_pose(Eigen::Matrix3d& K1, Eigen::Matrix3d& R1, Eigen::Vector3d& t1,
        Eigen::Matrix3d& K2,  Eigen::Matrix3d& R2, Eigen::Vector3d& t2,
        Eigen::Vector3d& p1, Eigen::Vector3d& p2);

// 像素坐标转相机的归一化坐标
Eigen::Vector3d  pix2cam(Eigen::Matrix3d& K, Eigen::Vector3d& pix);

// dlt 求解 基础矩阵
void find_fundamental_dlt(std::vector<Eigen::Vector3d>& imagePoints1,
        std::vector<Eigen::Vector3d>& imagePoints2, Eigen::Matrix3d& F);


// dlt 求解 本质矩阵
void find_essential_dlt(std::vector<Eigen::Vector3d>& camPoints1,
                        std::vector<Eigen::Vector3d>& camPoints2, Eigen::Matrix3d& E);

// 根据F矩阵，求解E矩阵
void find_essential_from_fundamental(Eigen::Matrix3d& K1,Eigen::Matrix3d& K2, Eigen::Matrix3d& F, Eigen::Matrix3d& E);


// E 矩阵分解
void decompose_from_essential(Eigen::Matrix3d& E, Eigen::Matrix3d& R1,Eigen::Matrix3d& R2,  Eigen::Vector3d& t1, Eigen::Vector3d& t2);


}
#endif //SLAM_ALGORITHM_STEREO_H
