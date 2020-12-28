//
// Created by atway on 2020/12/27.
//

#include "Stereo/stereo.h"
using namespace SLAM_STEREO;
using namespace cv;
using namespace Eigen;
int main() {

    std::string baseRoot="/home/atway/soft/opencv4.1/opencv-4.1.0/samples/data/";
    Mat leftImg = imread(baseRoot+"left01.jpg", cv::IMREAD_GRAYSCALE);
    Mat rightImg = imread(baseRoot+"right01.jpg", cv::IMREAD_GRAYSCALE);
    Size  patternSize(9, 6);


    // 相机内参
    Mat leftCameraMatrix, leftCameraDistCoeffs;
    Mat rightCameraMatrix, rightCameraDistCoeffs;

    FileStorage fs(baseRoot+"intrinsics.yml", FileStorage::READ);
    if( fs.isOpened() )
    {
        fs["M1"] >> leftCameraMatrix;
        fs["D1"] >> leftCameraDistCoeffs;
        fs["M2"] >> rightCameraMatrix;
        fs["D2"] >>  rightCameraDistCoeffs;
        fs.release();
    }
    else
        std::cout << "Error: can not load the intrinsic parameters\n";

    std::vector<Point2f> leftImgCorners, rightImgCorners;
    Mat drawLeftImg, drawRightImg;
    // 提取棋盘格角点
    SLAM_STEREO::findCorners(leftImg, patternSize, leftImgCorners,drawLeftImg);
    SLAM_STEREO::findCorners(rightImg, patternSize, rightImgCorners,drawRightImg);

    // 角点去畸变
    std::vector<Point2f> leftUndistImgCorners, rightUndistImgCorners;
    cv::undistortPoints(leftImgCorners, leftUndistImgCorners, leftCameraMatrix, leftCameraDistCoeffs);
    cv::undistortPoints(rightImgCorners, rightUndistImgCorners, rightCameraMatrix, rightCameraDistCoeffs);

    std::vector<Eigen::Vector3d> leftUndistImgCornersEigen, rightUndistImgCornersEigen;
    for (int i = 0; i < leftImgCorners.size(); ++i) {
        Point2f& pt1 = leftUndistImgCorners.at(i);
        Point2f& pt2 = rightUndistImgCorners.at(i);
        leftUndistImgCornersEigen.push_back(Eigen::Vector3d(pt1.x, pt1.y, 1.));
        rightUndistImgCornersEigen.push_back(Eigen::Vector3d(pt2.x, pt2.y, 1.));
    }

    /***********根据棋盘格像素点计算基础矩阵***********/
    {
        // DLT 计算Fundamental Matrix
        Matrix3d F;
        find_fundamental_dlt(leftUndistImgCornersEigen, rightUndistImgCornersEigen, F);
        std::cout << "fundamental:" <<  F << std::endl;

        // 验证x2^Fx1
        for(int i=0; i<leftImgCorners.size(); ++i){
            float d = rightUndistImgCornersEigen[i].transpose()*F*leftUndistImgCornersEigen[i];
            std::cout << i << " fundamental epipolar constraint: " << d << std::endl;
        }

    }

    std::vector<Vector3d> camPoints1, camPoints2;
    /***********根据棋盘格归一化相机坐标计算本质矩阵***********/
    {
        Eigen::Matrix3d E, K1, K2;
        cv::cv2eigen(leftCameraMatrix, K1);
        cv::cv2eigen(rightCameraMatrix, K2);


        // 像素坐标转换为归一化相机坐标
        for(int i=0; i<leftUndistImgCornersEigen.size(); ++i){
            camPoints1.push_back(pix2cam(K1, leftUndistImgCornersEigen[i]));
            camPoints2.push_back(pix2cam(K2, rightUndistImgCornersEigen[i]));
        }
        // DLT 计算Essential Matrix
        find_essential_dlt(camPoints1, camPoints2, E);
        std::cout << "essential:" << E<< std::endl;

        // 姿态分解，求R t
        Matrix3d R1, R2;
        Vector3d t1, t2;
        decompose_from_essential(E, R1, R2, t1, t2);

        // 左相机的外参为[I, 0] ， 右相机有4个可能[R1, t1]  [R2, t1] [R1, t2] [R2, t2]
        Matrix3d I = Matrix3d ::Identity();
        Vector3d t0 = Vector3d::Zero();

        // 根据三角化后的三维点，在相机的前面，即z>0, 可以统计，这里为了简单以某个点。
        {

            Matrix3d R;
            Vector3d t;
            {
                bool ok1 = choose_pose(K1, I, t0, K2, R1, t1, leftUndistImgCornersEigen[0], rightUndistImgCornersEigen[0]);
                bool ok2 = choose_pose(K1, I, t0, K2, R1, t2, leftUndistImgCornersEigen[0], rightUndistImgCornersEigen[0]);
                bool ok3 = choose_pose(K1, I, t0, K2, R2, t1, leftUndistImgCornersEigen[0], rightUndistImgCornersEigen[0]);
                bool ok4 = choose_pose(K1, I, t0, K2, R2, t2, leftUndistImgCornersEigen[0], rightUndistImgCornersEigen[0]);
                std::cout <<" ok1 " << ok1<<  " ok2 " << ok2 << " ok3 " << ok3<< " ok4 " << ok4<< std::endl;
                if(ok1) {
                    R = R1; t=t1;
                }else if(ok2){
                    R=R1; t=t2;
                }else if(ok3){
                    R=R2; t=t1;
                }else if(ok4){
                    R=R2; t=t2;
                }
            }

            // 校验
            Matrix3d txR = corss(t)*R;
            for(int i=0; i<camPoints1.size(); ++i){
                Vector3d& p1 = camPoints1[i];
                Vector3d& p2 = camPoints2[i];
                float d= p2.transpose()*txR*p1;
                std::cout << i << " E epipolar constraint: " << d << std::endl;
            }

        }

    }





    /***********根据相机左右外参计算相机之间的姿态***********/
    {
        float boardSize=25.0f;
        std::vector<cv::Point3f> objectPoints;
        for(int h=0; h<patternSize.height; ++h){
            for(int w=0; w<patternSize.width; ++w){
                objectPoints.push_back(cv::Point3f(w*boardSize, h*boardSize, 0.f));
            }
        }
        // 计算左右相机，相对与棋盘格的外参
        Mat Rl, tl, Rr, tr;
        cv::solvePnP(objectPoints, leftImgCorners, leftCameraMatrix, leftCameraDistCoeffs, Rl, tl);
        cv::solvePnP(objectPoints, rightImgCorners, rightCameraMatrix, rightCameraDistCoeffs, Rr, tr);

        // 左相机->右相机 R T
        Mat R, T;
        R = Rr*Rl.t();
        T = tr - R*tl;

        std::cout << "R: " << R << std::endl;
        std::cout <<"T: " << T.t() << std::endl;


        Mat tx = (Mat_<double>(3, 3) <<
                0, -T.at<double>(2), T.at<double>(1),
                T.at<double>(2), 0, -T.at<double>(0),
                -T.at<double>(1), T.at<double>(0), 0);
        Mat E = tx*R;


        Eigen::Matrix3d txR;
        cv::cv2eigen(E, txR);
        for(int i=0; i<camPoints1.size(); ++i){
            Vector3d& p1 = camPoints1[i];
            Vector3d& p2 = camPoints2[i];
            float d= p2.transpose()*txR*p1;
            std::cout << i << " E epipolar constraint: " << d << std::endl;
        }

    }


    return 0;
}