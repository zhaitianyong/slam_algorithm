//
// Created by atway on 2020/7/13.
//

#ifndef SLAM_ALGORITHM_MY_EPNP_H
#define SLAM_ALGORITHM_MY_EPNP_H
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/Geometry>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace Eigen;

class My_EPNP {

public:
    My_EPNP(Eigen::Matrix3d &K, Eigen::MatrixXd& p3d, Eigen::MatrixXd& p2d);

    void estimate(Matrix3d& R, Vector3d& t);

private:

    void choose_control_points(std::vector<Eigen::Vector3d>& control_pts_world); // 选择控制点

    void compute_barycentric_coordinates(vector<Vector3d>& control_pts_world, vector<Vector4d>& hb); // 计算质心坐标

    void  camera_position_from_control_points(vector<Vector4d>& hb, vector<Vector3d>& control_pts_world, Matrix3d& R, Vector3d& t);

    void computeRt(MatrixXd& p3d_c, Matrix3d& R, Vector3d& t);

    void computeL6_10(const Eigen::MatrixXd& U, Eigen::MatrixXd& L6_10);

    void computeRho(Eigen::VectorXd& rho, vector<Vector3d>& p3d_w);

    void find_betas_approx_4(Eigen::MatrixXd L6_10, Eigen::VectorXd& rho,
                             Vector4d&  betas);
    void find_betas_approx_2(Eigen::MatrixXd L6_10, Eigen::VectorXd& rho,
                             Vector4d&  betas);
    void find_betas_approx_3(Eigen::MatrixXd L6_10, Eigen::VectorXd& rho,
                             Vector4d&  betas);

    void gauss_newton(Eigen::MatrixXd L6_10, Eigen::VectorXd& rho,
                      Vector4d& betas);

    double calcCameraPoints(Vector4d& betas,  MatrixXd& eigen_vectors, vector<Vector4d>& hb, MatrixXd& vp3d_camera);
    double reprojectError(Matrix3d& R, Vector3d& t);


    Eigen::MatrixXd vp3d;
    Eigen::MatrixXd vp2d;
    Eigen::Matrix3d K;
    double fx, fy, cx, cy;
};


#endif //SLAM_ALGORITHM_MY_EPNP_H
