//
// Created by atway on 2020/7/8.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/Geometry>
#include <opencv2/features2d.hpp>
#include <sophus/se3.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>

using namespace std;
using namespace Eigen;
using namespace cv;

void triangulate(MatrixXd& T1, MatrixXd& T2, Vector3d& pt1, Vector3d& pt2, Vector3d& p){
    // 3 * 4
    MatrixXd A(4, 4);
    A.array().row(0) = pt1(0) * T1.row(2) - T1.row(0);
    A.array().row(1) = pt1(1) * T1.row(2) - T1.row(1);
    A.array().row(2) = pt2(0) * T2.row(2) - T2.row(0);
    A.array().row(3) = pt2(1) * T2.row(2) - T2.row(1);

    JacobiSVD<MatrixXd> solver(A, ComputeThinU | ComputeThinV);
    // Matrix3d U = solver.matrixU();
    Matrix4d V =solver.matrixV();

    Vector4d x = V.rightCols(1);

    p(0) = x(0)/x(3);
    p(1) = x(1)/x(3);
    p(2) = x(2)/x(3);


}


// 归一化点到中心为0 半径为0.5 - 0.5
void normalize_points(vector<Point2f>& points, vector<cv::Point2f>& normalizedPoints, Matrix3d &T)
{
    float meanX = 0;
    float meanY = 0;
    const int N = points.size();

    normalizedPoints.resize(N);

    for(int i=0; i<N; i++)
    {
        meanX += points[i].x;
        meanY += points[i].y;
    }

    meanX = meanX/N;
    meanY = meanY/N;

    float meanDevX = 0;
    float meanDevY = 0;

    for(int i=0; i<N; i++)
    {
        normalizedPoints[i].x =  points[i].x; - meanX;
        normalizedPoints[i].y =  points[i].y; - meanY;

        meanDevX += fabs(normalizedPoints[i].x);
        meanDevY += fabs(normalizedPoints[i].y);
    }

    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    for(int i=0; i<N; i++)
    {
        normalizedPoints[i].x = normalizedPoints[i].x * sX;
        normalizedPoints[i].y = normalizedPoints[i].y * sY;
    }

    T << sX, 0.0, -meanX*sX,
         0.0, sY, -meanX*sY,
         0.0, 0.0, 1.0;
}

/**
 *
 * @param p 内点的概率
 * @param k 需要的样本个数
 * @param z 成功的概率  1-z = (1 - p^k)^N
 * @return 迭代次数 N
 */
int ransac_iterations(double p, int k, double z=0.99){
    return  log(1-z) / log(1 - pow(p, k));
}

void find_matches(Mat& img1, Mat& img2, vector<Point2f>& imagePoints1, vector<Point2f>& imagePoints2){

    //
    Ptr<FeatureDetector> detector = ORB::create();
    //Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;

    detector->detectAndCompute(img1, noArray(), kpts1, desc1);
    detector->detectAndCompute(img2, noArray(), kpts2, desc2);

    //
    vector<DMatch> matches;
    matcher->match(desc1, desc2, matches);
    auto min_max = minmax_element(matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2){return m1.distance < m2.distance;});


    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    vector<DMatch> goodMatches;
    for (int i =0; i< matches.size(); i++){

        if(matches[i].distance <=max(2*min_dist, 30.)){
            goodMatches.push_back(matches[i]);
        }
    }
    Mat out;
    drawMatches(img1, kpts1, img2, kpts2, goodMatches, out);

    imshow("out", out);
    cv::waitKey(1000);
    cv::destroyAllWindows();
    if (goodMatches.size()< 20)
        return;

    for(auto & md : goodMatches){
        Point2f& pt1 = kpts1[md.queryIdx].pt;
        Point2f& pt2 = kpts2[md.trainIdx].pt;
        imagePoints1.push_back(kpts1[md.queryIdx].pt);
        imagePoints2.push_back(kpts1[md.trainIdx].pt);
    }
}

// 归一化图像坐标
void normal_image_points(vector<Point2f>& imagePoints, Mat& img){

    int width = img.cols;
    int height = img.rows;
    int max_ = max(width, height);

    for(auto& pt : imagePoints){
        pt.x = (pt.x + 0.5 - width/2.0)/ max_;
        pt.y = (pt.y + 0.5 - height/2.0)/ max_;
    }
}

void essential_matrix_from_fundamental(Matrix3d& F, Mat& K, Matrix3d& E){
    // E = K.-T * F × K

    Matrix3d K_;
    cv::cv2eigen(K, K_);

    E = (K_.inverse()).transpose() * F * K_.transpose();

    JacobiSVD<MatrixXd> solver (E, ComputeThinU | ComputeThinV);
    Matrix3d U = solver.matrixU();
    Matrix3d V = solver.matrixV();
    Vector3d vsigma = solver.singularValues();

    double sigma = (vsigma(0) + vsigma(1))/2.;
    Matrix3d sigma_E = Matrix3d ::Zero();
    sigma_E(0, 0) = sigma;
    sigma_E(1, 1) = sigma;

    E = U*sigma_E*V.transpose();
}

// F 基础矩阵是秩为2、自由度为7的齐次矩阵
void fundamental_matrix_point(vector<Point2f>& imagePoints1, vector<Point2f>& imagePoints2, Matrix3d& F){
    int N = imagePoints1.size();
    MatrixXd H(N, 9);
    for (int i=0; i<N; i++){
        double x1 = imagePoints1[i].x;
        double y1 = imagePoints1[i].y;

        double x2 = imagePoints2[i].x;
        double y2 = imagePoints2[i].y;
        //
        H(i, 0) = x2*x1;
        H(i, 1) = x2*y1;
        H(i, 2) = x2;
        H(i, 3) = y2*x1;
        H(i, 4) = y2*y1;
        H(i, 5) = y2;
        H(i, 6) = x1;
        H(i, 7) = y1;
        H(i, 8) = 1.;
    }

    JacobiSVD<MatrixXd> solver_H( H, ComputeThinU | ComputeThinV);
    MatrixXd  V_H = solver_H.matrixV();
    VectorXd v = V_H.rightCols(1); // 最后1列
    // 分解
    F <<    v(0), v(1), v(2),
            v(3), v(4), v(5),
            v(6), v(7), v(8);

    // 对E进行分解
    JacobiSVD<MatrixXd> solver_F(F, ComputeThinU|ComputeThinV);
    Matrix3d U_F = solver_F.matrixU();
    Matrix3d V_F = solver_F.matrixV();
    Vector3d vsigma = solver_F.singularValues();


    Matrix3d sigma_E = Matrix3d::Zero();
    sigma_E(0, 0) = vsigma(0);
    sigma_E(1, 1) = vsigma(1);
    F = U_F * sigma_E * V_F.transpose();
    //F /=  F(2, 2);

}


double distance_l2(Vector3d& p1, Vector3d& p2, Matrix3d& F, double sigma){
    // 直线方程
    Eigen::Vector3d l2 = F*p1;
    //Eigen::Vector3d l1 = p2.transpose()*F_;
    // 点到直线的距离
    double sum1 = p2.transpose() * l2;
    double squareDist1 = sqrt(sum1*sum1 / (l2(0) * l2(0) + l2(1)*l2(1)));

    double invSigmaSquare = 1.0 / (sigma*sigma);
    const float chiSquare1 = squareDist1*invSigmaSquare;

    return chiSquare1;
}
// Ransac 计算F矩阵
void fundamental_maxtrix_ransac(vector<Point2f>& imagePoints1, vector<Point2f>& imagePoints2, Matrix3d& F){

    int N = imagePoints1.size();

    // 归一化
    Matrix3d T1, T2;
    vector<Point2f> normalPoints1, normalPoints2;

    normalize_points(imagePoints1, normalPoints1, T1);
    normalize_points(imagePoints2, normalPoints2, T2);

    int maxIters = 1000;
    // 设置迭代次数
    // 通过概率公式计算
    int iters = ransac_iterations(0.5, 8);
    maxIters = min(iters, maxIters);

    cout << "iters = " << iters << endl;
    // 用于判读匹配对是否为内点
    const double inlier_thresh = 3.84;
    // ransac 最终估计的内点
    std::vector<int> best_inliers;
    for (int iter=0; iter<maxIters; iter++){

        std::set<int> indices;
        while(indices.size()<8){
            indices.insert( std::rand() % N);
        }
        vector<Point2f> pts1, pts2;
        std::set<int>::const_iterator index_iter = indices.cbegin();
        for(int i=0; i<8; i++, index_iter++){
            pts1.push_back(imagePoints1[*index_iter]);
            pts2.push_back(imagePoints2[*index_iter]);
        }
        Matrix3d  F21;
        fundamental_matrix_point(normalPoints1,normalPoints2,F21);

        // 恢复
        Matrix3d F21_ = T2.transpose()*F21*T1;
//        cout << iter << " " << F_ << endl;
        // 距离判断，内点个数统计
        {
            std::vector<int> inlier_indices;
            for(int i=0; i<N; i++){
                Eigen::Vector3d p1(imagePoints1[i].x, imagePoints1[i].y, 1.0);
                Eigen::Vector3d p2(imagePoints2[i].x, imagePoints2[i].y, 1.0);
//                double a = (p2.transpose()*F*p1).sum();
//                Eigen::Vector3d l2 = F_*p1;
//                Eigen::Vector3d l1 = p2.transpose()*F_;
//                double b = pow(l2(0), 2) + pow(l2(1), 2) + pow(l1(0), 2) + pow(l1(1), 2);
//                double r = a*a/b;
//                double thresh = 1. / (inlier_thresh*inlier_thresh);
                double r = distance_l2(p1, p2, F21_, 1.0);
                cout << r << endl;
                if(r < inlier_thresh){
                    inlier_indices.push_back(i);
                }
            }

            if(inlier_indices.size()>best_inliers.size()){
                best_inliers.swap(inlier_indices);
            }
        }
    }


    cout << "inlier size  = " << best_inliers.size() << endl;
    {
        if (best_inliers.size()<8)
            return;
        // 所有的内点 最小二乘法
        vector<Point2f> pts1, pts2;
        for(int i : best_inliers){
            pts1.push_back(imagePoints1[i]);
            pts2.push_back(imagePoints2[i]);
        }
        fundamental_matrix_point(pts1,pts2,F);
    }

}

void load_points_from_file(vector<Point2f>& pts1, vector<Point2f>& pts2){
    std::ifstream in("/home/atway/code/slam/MVGAlgorithm/slam_algorithm/BundleAdjustment/correspondences.txt");
    assert(in.is_open());

    std::string line, word;
    int n_line = 0;
    float x1,y1,x2,y2;
    while(getline(in, line)){

        std::stringstream stream(line);
        if(n_line==0){
            int n_corrs = 0;
            stream>> n_corrs;
//            pts1.resize(n_corrs);
//            pts2.resize(n_corrs);
            n_line ++;
            continue;
        }
        if(n_line>0){
            stream>>x1 >> y1 >> x2 >> y2;
            pts1.push_back(Point2f(x1, y1));
            pts2.push_back(Point2f(x2, y2));
        }
        n_line++;
    }
}

void decomposition_essential_matrix(Matrix3d& E, Matrix3d& R1, Matrix3d& R2, Vector3d& t1, Vector3d& t2){

    JacobiSVD<MatrixXd> solver(E, ComputeThinU | ComputeThinV);
    Matrix3d U = solver.matrixU();
    Matrix3d V =solver.matrixV();
    Vector3d vSigma = solver.singularValues();
    cout << "vSigma : " << vSigma.transpose() << endl;

    Matrix3d R_90, R_270 ;
    R_90 << 0., -1., 0,
            1., 0., 0.,
            0., 0., 1.;
    R_270 << 0., 1., 0.,
            -1., 0., 0.,
             0., 0., 1.;

    R1 = U*R_90*V.transpose();
    R2 = U*R_270*V.transpose();

    if (R1.determinant() < 0){
        R1 = -R1;
    }
    if (R2.determinant() < 0){
        R2 = -R2;
    }

    t1 = U.rightCols(1);
    t2 = -t1;
//    t1 = t1/ t1.norm();
//    t2 = -t1;
}

void get_pose_from_essential(Matrix3d& E,vector<Point2f>& pts1, vector<Point2f>& pts2,Mat& K){
    Matrix3d R1, R2;
    Vector3d t1, t2;
    decomposition_essential_matrix(E, R1,R2, t1, t2);
    cout << "t1 " << t1.transpose() << endl;
    cout << "t2 " << t1.transpose() << endl;

    // 组合 （R1, t1）(R1, t2). (R2, t1) (R2, t2)
    // 判断点的深度为正
    vector<pair<Matrix3d, Vector3d>> poses;
    poses.push_back(make_pair(R1, t1));
    poses.push_back(make_pair(R1, t2));
    poses.push_back(make_pair(R2, t1));
    poses.push_back(make_pair(R2, t2));

    double fx, fy, cx, cy;
    fx = K.at<double>(0, 0);
    fy = K.at<double>(1, 1);
    cx = K.at<double>(0, 2);
    cy = K.at<double>(1, 2);
    int N = pts1.size();
    int numbers[4] = { 0, 0, 0, 0};

    MatrixXd T1(3, 4);
    T1.block<3, 3>(0, 0) = Matrix3d ::Identity();
    T1.rightCols(1) << 0., 0., 0.;
    Vector3d O1(0., 0., 0.);
    Vector3d d1(0., 0., 1.0);
    for (int i=0; i<4; i++){
        pair<Matrix3d, Vector3d> pose=poses[i];
        MatrixXd T2(3, 4);
        T2.block<3, 3>(0, 0) = pose.first;
        T2.rightCols(1) = pose.second;
        Vector3d O2 = -pose.first.transpose()*pose.second;
        Vector3d d2 = pose.first.row(2);
        for (int j = 0; j < N; ++j) {
            Vector3d p1((pts1[j].x - cx)/fx, (pts1[j].y - cy)/fy, 1.0);
            Vector3d p2((pts2[j].x - cx)/fx, (pts2[j].y - cy)/fy, 1.0);
            Vector3d P=Vector3d ::Zero();
            triangulate(T1, T2, p1, p2, P);
            // 判断
            double z1 = (P-O1).transpose()*d1;
            double z2 = (P-O2).transpose()*d2;
            if(z1>0 && z2>0){
                //cout << i << "  " <<  T2 << endl;
                numbers[i] += 1;
            }
        }
    }
    for(auto& i : numbers)
        cout << i << " ";
    cout << endl;
}


int main(int argc, char** argv) {
    Mat img1 = imread("/home/atway/code/slam/slambook/slambook2/ch7/1.png");
    Mat img2 = imread("/home/atway/code/slam/slambook/slambook2/ch7/2.png");

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    vector<Point2f> imagePoints1, imagePoints2;
    //load_points_from_file(imagePoints1, imagePoints2);
    find_matches(img1, img2, imagePoints1, imagePoints2);
    cout << "match size: " << imagePoints1.size() << endl;

    if(false)
    {
        //-- 计算基础矩阵
        Mat fundamental_matrix;
        fundamental_matrix = findFundamentalMat(imagePoints1, imagePoints2, CV_FM_8POINT);
        cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;

        //-- 计算本质矩阵
        Point2d principal_point(325.1, 249.7);  //相机光心, TUM dataset标定值
        double focal_length = 521;      //相机焦距, TUM dataset标定值
        Mat essential_matrix;
        essential_matrix = findEssentialMat(imagePoints1, imagePoints2, focal_length, principal_point);
        cout << "essential_matrix is " << endl << essential_matrix << endl;


        //-- 计算单应矩阵
        //-- 但是本例中场景不是平面，单应矩阵意义不大
        Mat homography_matrix;
        homography_matrix = findHomography(imagePoints1, imagePoints2, RANSAC, 3);
        cout << "homography_matrix is " << endl << homography_matrix << endl;

        //-- 从本质矩阵中恢复旋转和平移信息.
        // 此函数仅在Opencv3中提供
        Mat R, t;
        recoverPose(essential_matrix, imagePoints1, imagePoints2, R, t, focal_length, principal_point);
        cout << "R is " << endl << R << endl;
        cout << "t is " << endl << t << endl;

    }
    if(true){
        Eigen::Matrix3d  essential_matrix, fundamental_matrix;
        fundamental_maxtrix_ransac(imagePoints1, imagePoints2, fundamental_matrix);
        cout << "fundamental_matrix  is " << endl << fundamental_matrix << endl;
        essential_matrix_from_fundamental(fundamental_matrix, K, essential_matrix);
        cout << "essential_matrix  is " << endl << essential_matrix << endl;
        get_pose_from_essential(essential_matrix, imagePoints1, imagePoints2, K);

    }
    /*
    {
        Matrix3d F;
        fundamental_maxtrix_ransac(imagePoints1, imagePoints2, F);
        cout << "F: " << F << endl;
    }
     */

    return 0;
}