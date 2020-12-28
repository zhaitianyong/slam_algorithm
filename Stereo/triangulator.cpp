//
// Created by atway on 2020/12/14.
//
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/core/eigen.hpp>
using namespace cv;
using namespace Eigen;

/**
 *
 * @param T1 相机O1的投影矩阵
 * @param T2 相机O2的投影矩阵
 * @param p1 相机O1的图像的像素点
 * @param p2 相机O2的图像的像素点
 * @param out 输出的三维坐标
 */
void triangulatePoint(Matrix<double, 3, 4>& T1, Matrix<double, 3, 4>& T2, Vector2d& p1, Vector2d&p2, Vector3d& out){
    // 构建Ax = 0;
    Matrix<double, 4, 4> A;
    A.row(0) = p1(0)*T1.row(2) - T1.row(0);
    A.row(1) = p1(1)*T1.row(2) - T1.row(1);
    A.row(2) = p2(0)*T2.row(2) - T2.row(0);
    A.row(3) = p2(1)*T2.row(2) - T2.row(1);

    // SVD 分解
    Eigen::JacobiSVD<MatrixXd> solver(A, Eigen::ComputeFullU|Eigen::ComputeThinV);
    MatrixXd V = solver.matrixV();
    VectorXd X = V.rightCols(1);

    out[0] = X(0)/X(3);
    out[1] = X(1)/X(3);
    out[2] = X(2)/X(3);
}

void triangulateByOpenCV(Matrix<double, 3, 4>& T1, Matrix<double, 3, 4>& T2, Vector2d& p1, Vector2d&p2, Vector3d& out){

    Mat T_mat1, T_mat2;
    cv::eigen2cv(T1, T_mat1);
    cv::eigen2cv(T2, T_mat2);


    std::vector<Point2d> p1s,p2s;
    p1s.push_back(Point2d(p1.x(), p1.y()));
    p2s.push_back(Point2d(p2.x(), p2.y()));


    /*
     *  @param projMatr1 3x4 projection matrix of the first camera.
        @param projMatr2 3x4 projection matrix of the second camera.
        @param projPoints1 2xN array of feature points in the first image. In case of c++ version it can
        be also a vector of feature points or two-channel matrix of size 1xN or Nx1.
        @param projPoints2 2xN array of corresponding points in the second image. In case of c++ version
        it can be also a vector of feature points or two-channel matrix of size 1xN or Nx1.
        @param points4D 4xN array of reconstructed points in homogeneous coordinates.
     *
     */
    Mat pts_4d;
    cv::triangulatePoints(T_mat1, T_mat2, p1s, p2s, pts_4d);

    Mat X = pts_4d.col(0);
    X /=X.at<double>(3,0);

    out[0] = X.at<double>(0, 0);
    out[1] = X.at<double>(1, 0);
    out[2] = X.at<double>(2, 0);

}

int main(){

    Matrix<double, 3, 4> T1, T2;
    T1(0, 0) = 0.919653;    T1(0, 1)=-0.000621866; T1(0, 2)= -0.00124006; T1(0, 3) = 0.00255933;
    T1(1, 0) = 0.000609954; T1(1, 1)=0.919607    ; T1(1, 2)= -0.00957316; T1(1, 3) = 0.0540753;
    T1(2, 0) = 0.00135482;  T1(2, 1) =0.0104087  ; T1(2, 2)= 0.999949;    T1(2, 3) = -0.127624;

    T2(0, 0) = 0.920039;    T2(0, 1)=-0.0117214;  T2(0, 2) = 0.0144298;   T2(0, 3)   = 0.0749395;
    T2(1, 0) = 0.0118301;   T2(1, 1)=0.920129  ;  T2(1, 2) = -0.00678373; T2(1, 3) = 0.862711;
    T2(2, 0) = -0.0155846;  T2(2, 1) =0.00757181; T2(2, 2) = 0.999854 ;   T2(2, 3)   = -0.0887441;

    Vector2d p1, p2;
    p1(0) = 0.289986; p1(1) = -0.0355493;
    p2(0) = 0.316154; p2(1) =  0.0898488;



    Vector3d out1, out2;
    triangulatePoint(T1, T2, p1, p2, out1);

    triangulateByOpenCV(T1, T2, p1, p2, out2);

    std::cout<< "result 1 >> " << out1.transpose() << std::endl;
    std::cout<< "result 2 >> " << out2.transpose() << std::endl;

    return 0;
}
