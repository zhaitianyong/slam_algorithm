//
// Created by atway on 2020/12/21.
//

#include "stereo.h"
#include<Eigen/Eigenvalues>
#include <set>
namespace SLAM_STEREO
{
    Eigen::Matrix3d corss(Eigen::Vector3d& t){

        Eigen::Matrix3d A;
        A << 0, -t(2), t(1), t(2), 0, -t(0), -t(1), t(0), 0;

        return A;
    }

    bool findCorners(cv::Mat& img, cv::Size& patternSize, std::vector<cv::Point2f> &corners,cv::Mat& drawMat){
        bool ok = cv::findChessboardCorners(img, patternSize, corners);
        if(ok){
            cornerSubPix(img, corners, cv::Size(11,11), cv::Size(-1,-1),
                         cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
                                      30, 0.01));
        }

        cv::cvtColor(img, drawMat, cv::COLOR_GRAY2BGR);
        cv::drawChessboardCorners(drawMat, patternSize, corners, ok);
        return ok;
    }
    // 这里的K1 为左相机的内参， R1为左相机的旋转矩阵 也就是单位矩阵， t1为左相机的平移向量，这里为[0, 0, 0]
    // K2 为右相机的内参， R2 为计算的两个可能的R， t1 为计算的两个可能的t​
    bool choose_pose(Eigen::Matrix3d& K1, Eigen::Matrix3d& R1, Eigen::Vector3d& t1,
            Eigen::Matrix3d& K2,  Eigen::Matrix3d& R2, Eigen::Vector3d& t2,
            Eigen::Vector3d& p1, Eigen::Vector3d& p2){

        // 选择合适的
        Eigen::Matrix<double, 3, 4> P1, P2;
        P1 =  Eigen::Matrix<double, 3, 4>::Zero();
        P2 =  Eigen::Matrix<double, 3, 4>::Zero();

        P1.array().block(0, 0, 3, 3) = R1;
        P1.array().block(0, 3, 3, 1) = t1;
        P1 = K1*P1;

        P2.array().block(0, 0, 3, 3) = R2;
        P2.array().block(0, 3, 3, 1) = t2;
        P2 = K2*P2;

        Eigen::Vector3d out;
        triangulatePoint(P1, P2, p1, p2, out);
        Eigen::Vector3d cam =  R2*out+t2;


        if(out(2)>0 && cam(2)>0)
            return true;
        else
            return false;

    }

    void triangulatePoint(Eigen::Matrix<double, 3, 4>& T1, Eigen::Matrix<double, 3, 4>& T2, Eigen::Vector3d& p1, Eigen::Vector3d&p2, Eigen::Vector3d& out){
        // 构建Ax = 0;
        Eigen::Matrix<double, 4, 4> A;
        A.row(0) = p1(0)*T1.row(2) - T1.row(0);
        A.row(1) = p1(1)*T1.row(2) - T1.row(1);
        A.row(2) = p2(0)*T2.row(2) - T2.row(0);
        A.row(3) = p2(1)*T2.row(2) - T2.row(1);

        // SVD 分解
        Eigen::JacobiSVD<Eigen::MatrixXd> solver(A, Eigen::ComputeFullU|Eigen::ComputeThinV);
        Eigen::MatrixXd V = solver.matrixV();
        Eigen::VectorXd X = V.rightCols(1);

        out[0] = X(0)/X(3);
        out[1] = X(1)/X(3);
        out[2] = X(2)/X(3);
    }
    void find_fundamental_dlt(std::vector<Eigen::Vector3d>& imagePoints1,
                              std::vector<Eigen::Vector3d>& imagePoints2, Eigen::Matrix3d& F){

        assert(imagePoints2.size()==imagePoints1.size());
        using namespace Eigen;

        int N = imagePoints1.size();
        MatrixXd A(N, 9);

        for (int i = 0; i <N; ++i) {
            Vector3d& p1 =imagePoints1.at(i);
            Vector3d& p2 =imagePoints2.at(i);

            A(i, 0) = p1[0]*p2[0];
            A(i, 1) = p1[1]*p2[0];
            A(i, 2) = p2[0];
            A(i, 3) = p1[0]*p2[1];
            A(i, 4) = p1[1]*p2[1];
            A(i, 5) = p2[1];
            A(i, 6) = p1[0];
            A(i, 7) = p1[1];
            A(i, 8) = 1.0;
        }
        // SVD 分解
        Eigen::JacobiSVD<MatrixXd> solver(A, Eigen::ComputeFullU|Eigen::ComputeFullV);
        MatrixXd V = solver.matrixV();
        VectorXd X = V.col(8);

        Matrix3d F_;
        F_ << X(0), X(1), X(2), X(3), X(4), X(5), X(6), X(7), X(8);
        Eigen::JacobiSVD<Matrix3d> solverF(F_, Eigen::ComputeFullU|Eigen::ComputeFullV);
        Matrix3d V_ = solverF.matrixV();
        Matrix3d U_ = solverF.matrixU();
        Vector3d sig_ = solverF.singularValues();
        Matrix3d S_=Matrix3d::Zero();
        S_(0, 0) = sig_(0);
        S_(1, 1) = sig_(1);
        S_(2, 2) = 0.0;
        F = U_ * S_* V_.transpose();
        F.array() /= F(2, 2);
    }

    Eigen::Vector3d  pix2cam(Eigen::Matrix3d& K, Eigen::Vector3d& pix){

        Eigen::Vector3d cam = K.inverse()*pix;

        cam.array() /= cam(2);


        return cam;
    }

    void find_essential_dlt(std::vector<Eigen::Vector3d>& camPoints1,
                            std::vector<Eigen::Vector3d>& camPoints2, Eigen::Matrix3d& E){


        assert(camPoints1.size()==camPoints2.size());
        using namespace Eigen;

        int N = camPoints1.size();
        MatrixXd A(N, 9);

        for (int i = 0; i <N; ++i) {
            Vector3d& p1 =camPoints1.at(i);
            Vector3d& p2 =camPoints2.at(i);

            A(i, 0) = p1[0]*p2[0];
            A(i, 1) = p1[1]*p2[0];
            A(i, 2) = p2[0];
            A(i, 3) = p1[0]*p2[1];
            A(i, 4) = p1[1]*p2[1];
            A(i, 5) = p2[1];
            A(i, 6) = p1[0];
            A(i, 7) = p1[1];
            A(i, 8) = 1.0;
        }
        // SVD 分解
        Eigen::JacobiSVD<MatrixXd> solver(A, Eigen::ComputeFullU|Eigen::ComputeFullV);
        MatrixXd V = solver.matrixV();
        VectorXd X = V.col(8);

        Matrix3d E_;
        E_ << X(0), X(1), X(2), X(3), X(4), X(5), X(6), X(7), X(8);
        Eigen::JacobiSVD<Matrix3d> solverF(E_, Eigen::ComputeFullU|Eigen::ComputeFullV);
        Matrix3d V_ = solverF.matrixV();
        Matrix3d U_ = solverF.matrixU();
        Vector3d sig_ = solverF.singularValues();
        Matrix3d S_=Matrix3d::Zero();
        double s = (sig_(0)+sig_(1))/2;
        S_(0, 0) = s;
        S_(1, 1) = s;
        S_(2, 2) = 0;
        E = U_ * S_* V_.transpose();
    }


    void find_essential_from_fundamental(Eigen::Matrix3d& K1,Eigen::Matrix3d& K2, Eigen::Matrix3d& F, Eigen::Matrix3d& E){
        using namespace Eigen;

        Matrix3d E_ = K2.transpose()*F*K1;
        //E_.array() /= E_(2, 2);
        Eigen::JacobiSVD<Matrix3d> solverF(E_, Eigen::ComputeFullU|Eigen::ComputeFullV);
        Matrix3d V_ = solverF.matrixV();
        Matrix3d U_ = solverF.matrixU();
        Vector3d sig_= solverF.singularValues();
        Matrix3d S_ =Matrix3d::Zero();
        double s = (sig_(0)+sig_(1))/2;
        S_(0, 0) = s;
        S_(1, 1) = s;
        S_(2, 2) = 0;

        E = U_ * S_* V_.transpose();


    }

    void decompose_from_essential(Eigen::Matrix3d& E, Eigen::Matrix3d& R1,Eigen::Matrix3d& R2,  Eigen::Vector3d& t1, Eigen::Vector3d& t2){

        using namespace Eigen;
        // 绕z轴旋转90度和旋转-90度
        Matrix3d R_90, R_90_;
        R_90 << 0, -1, 0, 1, 0, 0, 0, 0, 1;
        R_90_ << 0, 1, 0, -1, 0, 0, 0, 0, 1;

        //svd 分解E
        Eigen::JacobiSVD<Matrix3d> solver(E, Eigen::ComputeFullU|Eigen::ComputeFullV);
        Matrix3d U = solver.matrixU();
        Matrix3d V = solver.matrixV();
        Vector3d sig = solver.singularValues();

        Matrix3d S=Matrix3d::Zero();
        S(0, 0) = sig(0);
        S(1, 1) = sig(1);
        S(2, 2) = sig(2);


        //  det(E)= 1
        if(U.determinant()<0.0){
            for(int i=0; i<3; ++i)
                U(i, 2) = -U(i, 2);
        }
        if(V.determinant()<0.0){
            for(int i=0; i<3; ++i)
                V(i, 2) = -V(i, 2);
        }

        R1 = U*R_90*V;
        R2 = U*R_90_*V;

        t1 = U.col(2);
        t2 = -U.col(2);

        // 单位化
        double m10 = sqrt(R1.col(0).transpose()*R1.col(0));
        double m11 = sqrt(R1.col(1).transpose()*R1.col(1));
        double m12 = sqrt(R1.col(2).transpose()*R1.col(2));

        R1.array().col(0)  /= m10;
        R1.array().col(1)  /= m11;
        R1.array().col(2)  /= m12;

        double m20 = sqrt(R2.col(0).transpose()*R2.col(0));
        double m21 = sqrt(R2.col(1).transpose()*R2.col(1));
        double m22 = sqrt(R2.col(2).transpose()*R2.col(2));

        R2.array().col(0)  /= m20;
        R2.array().col(1)  /= m21;
        R2.array().col(2)  /= m22;

        double mt1 = sqrt(t1.transpose()*t1);
        t1.array() /= mt1;

        double mt2 = sqrt(t2.transpose()*t2);
        t2.array() /= mt2;

    }
}