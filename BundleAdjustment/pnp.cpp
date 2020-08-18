//
// Created by atway on 2020/7/7.
//

#include <iostream>


#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/Geometry>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>

#include <sophus/se3.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/problem.h>

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include "BundleAdjustment/My_EPNP.h"
using namespace std;
using namespace Eigen;
using namespace cv;


class PROJECT_COST{

public:
    PROJECT_COST(double* K, Point2f& imagePoint, Point3f& objectPoint){
        imagePoint_ = imagePoint;
        objectPoint_ = objectPoint;
        K_ = K;
    }

    template<typename T>
    bool operator()(
            const T *camera,
            T *residuals)const {
        T point[3] = {T(objectPoint_.x), T(objectPoint_.y), T(objectPoint_.z)};
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);

        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];


        T x_ = T(K_[0]) * p[0] / p[2] + T(K_[2]);
        T y_ = T(K_[1]) * p[1] / p[2] + T(K_[3]);

        residuals[0] = T(imagePoint_.x) - x_;
        residuals[1] = T(imagePoint_.y) - y_;

        return true;
    }
    static ceres::CostFunction* Create(double* K, Point2f& imagePoint, Point3f& objectPoint) {
        return (new ceres::AutoDiffCostFunction<PROJECT_COST, 2, 6>(
                new PROJECT_COST(K, imagePoint, objectPoint)));
    }

private:
    Point2f imagePoint_;
    Point3f objectPoint_;
    double* K_; // 相机内参
};


class VertexSE3Pose : public g2o::BaseVertex<6, Sophus::SE3d>{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexSE3Pose(){};

    bool read(std::istream& is) override {};

    bool write(std::ostream& os) const override {};

    virtual void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }

    virtual void oplusImpl(const double* update) override {
        Eigen::VectorXd update_(6);
        update_ << update[0], update[1], update[2], update[3],update[4],update[5];
        //setEstimate(SE3Quat::exp(update)*estimate());
        _estimate = Sophus::SE3d::exp(update_) * _estimate;
    }
};


class EdgeProject: public  g2o::BaseUnaryEdge<2, Eigen::Vector2d , VertexSE3Pose>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProject(const Eigen::Vector3d& pos, const Eigen::Matrix3d& K){
        pos_ = pos;
        K_ = K;
    }

    bool read(std::istream& is) override {};

    bool write(std::ostream& os) const override {};

    virtual void computeError() override {
        // 计算误差
        const VertexSE3Pose *v = static_cast<VertexSE3Pose*>(_vertices[0]);
        Sophus::SE3d T = v->estimate(); // 拿到节点的姿态
        Eigen::Vector3d pix =  K_ * (T * pos_);
        pix = pix / pix(2);
        _error = _measurement - pix.head(2);

    }

    virtual void linearizeOplus() override {
        // 更新
        const VertexSE3Pose *v = static_cast<VertexSE3Pose*>(_vertices[0]);
        Sophus::SE3d T = v->estimate(); // 拿到节点的姿态
        Eigen::Vector3d pc =  T * pos_; // 相机坐标系

        // 定义雅克比矩阵
        double fx, fy, cx, cy;
        fx = K_(0, 0);
        fy = K_(1, 1);
        cx = K_(0, 2);
        cy = K_(1, 2);
        double inv_z = 1.0 / pc[2];
        double inv_z2 = inv_z * inv_z;

        _jacobianOplusXi << -fx * inv_z,
            0,
            fx * pc[0] * inv_z2,
            fx * pc[0] * pc[1] * inv_z2,
            -fx - fx * pc[0] * pc[0] * inv_z2,
            fx * pc[1] * inv_z,
            0,
            -fy * inv_z,
            fy * pc[1] * inv_z2,
            fy + fy * pc[1] * pc[1] * inv_z2,
            -fy * pc[0] * pc[1] * inv_z2,
            -fy * pc[0] * inv_z;
    }

private:
    Eigen::Vector3d pos_;
    Eigen::Matrix3d K_;
};


/**
 * 旋转矩阵转换为旋转向量
 */
Vector3d rotationMatrix2Vector(const Matrix3d& R)
{

    AngleAxisd r;
    r.fromRotationMatrix(R);
    return r.angle()*r.axis();
}
/**
 *
 * 旋转向量到旋转矩阵
 */

Matrix3d rotationVector2Matrix(const Vector3d& v)
{


    double s = sqrt(v.dot(v));
    Vector3d axis = v/s;
    AngleAxisd r( s, axis);

    return r.toRotationMatrix();
}

// 直接线性变换
// 求解的结果不好
void pnp_dlt(vector<Point2f>& imagePoints, vector<Point3f>& objectPoints, Mat& K, Sophus::SE3d& T){
    vector<Point2f> camNormalPoints;
    int N = imagePoints.size();
    double fx, fy, cx, cy;
    fx = K.at<double>(0, 0);
    fy = K.at<double>(1, 1);
    cx = K.at<double>(0, 2);
    cy = K.at<double>(1, 2);
    for(auto& pt: imagePoints){
        double x = ( pt.x - cx )/fx;
        double y = (pt.y - cy )/ fy;
        camNormalPoints.push_back(Point2f(x, y));
    }

    // x, y, z, 1, 0, 0, 0, 0, -ux, -uy, -uz, -u
    // 0, 0, 0, 0, x, y, z, 1, -vx, -vy, -vz, -v
    Eigen::MatrixXd H(2*N,12);

    for (int i=0; i<N; i++){

        H(2*i, 0) = objectPoints[i].x;
        H(2*i, 1) = objectPoints[i].y;
        H(2*i, 2) = objectPoints[i].z;
        H(2*i, 3) = 1.0;
        H(2*i, 4) = 0.0;
        H(2*i, 5) = 0.0;
        H(2*i, 6) = 0.0;
        H(2*i, 7) = 0.0;
        H(2*i, 8) = -camNormalPoints[i].x*objectPoints[i].x;
        H(2*i, 9) = -camNormalPoints[i].x*objectPoints[i].y;
        H(2*i, 10) = -camNormalPoints[i].x*objectPoints[i].z;
        H(2*i, 11) = -camNormalPoints[i].x;

        H(2*i+1, 0) = 0.0;
        H(2*i+1, 1) = 0.0;
        H(2*i+1, 2) = 0.0;
        H(2*i+1, 3) = 0.0;
        H(2*i+1, 4) = objectPoints[i].x;
        H(2*i+1, 5) = objectPoints[i].y;
        H(2*i+1, 6) = objectPoints[i].z;
        H(2*i+1, 7) = 1.0;
        H(2*i+1, 8) = -camNormalPoints[i].y*objectPoints[i].x;
        H(2*i+1, 9) = -camNormalPoints[i].y*objectPoints[i].y;
        H(2*i+1, 10) = -camNormalPoints[i].y*objectPoints[i].z;
        H(2*i+1, 11) = -camNormalPoints[i].y;
    }

    Matrix3d rMatrix;
    Vector3d tvec;
    {
        // svd 分解
        // 3. SVD分解
        JacobiSVD<MatrixXd> svdSolver ( H, ComputeThinU | ComputeThinV );
        Eigen::MatrixXd V = svdSolver.matrixV();

        VectorXd v = V.rightCols(1);


        rMatrix <<  v(0), v(1), v(2),
                v(4), v(5), v(6),
                v(8), v(9), v(10);
        tvec << v(3), v(7), v(11);
    }


    //cout << "v = " << v.transpose() << endl;

    // 矩阵计算
    // 分解：
    Matrix3d R;
    {
        JacobiSVD<MatrixXd> svdSolver (rMatrix, ComputeFullU | ComputeFullV);
        Eigen::Matrix3d V = svdSolver.matrixV();
        Eigen::Matrix3d U = svdSolver.matrixU();
        Eigen::Vector3d sigma_v = svdSolver.singularValues();
        Eigen::Matrix3d E = Matrix3d::Identity();

        double factor = 1.0/ sigma_v.mean();
        cout << "factor: " << factor << endl;
        R = U*E*V.transpose();
        tvec = factor*tvec;


        // Check + -
        int num_positive = 0;
        int num_negative = 0;
        for ( int i = 0; i < N ; i ++ ) {
            const double& x = objectPoints[i].x;
            const double& y = objectPoints[i].y;
            const double& z = objectPoints[i].z;

            double lambda = R(2, 0) * x + R(2, 1)*y + R(2, 2)*z + tvec(2);
            if ( lambda >= 0 ) {
                num_positive ++;
            } else {
                num_negative ++;
            }
        }
        cout << "num_positive " << num_positive << endl;
        cout << "num_negative "  << num_negative << endl;
        if ( num_positive < num_negative ) {
            R = -R;
            tvec = -tvec;
        }

    }
//    cout <<"R " << R << endl;
//    cout << "tvec:" << tvec.transpose() << endl;

    T = Sophus::SE3d(R, tvec);
}

// 优化算法
void pnp_gauss_newton(vector<Point2f>& imagePoints, vector<Point3f>& objectPoints, Mat& K, Sophus::SE3d& T){
    typedef Eigen::Matrix<double, 6, 1> Vector6d;

    vector<Eigen::Vector3d> points_3d;
    for(auto& pt: objectPoints){
        points_3d.push_back(Vector3d(pt.x, pt.y, pt.z));
    }

    int iters = 100;
    int N = imagePoints.size();
    double fx, fy, cx, cy;
    fx = K.at<double>(0, 0);
    fy = K.at<double>(1, 1);
    cx = K.at<double>(0, 2);
    cy = K.at<double>(1, 2);


    double lastCost=0;
    for (int iter=0; iter<iters; iter++){
        //
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();


        double cost=0;

        for (int i=0; i<N; i++){
            Eigen::Vector3d pc = T * points_3d[i];
            double inv_z = 1.0 / pc[2];
            double inv_z2 = inv_z * inv_z;
            Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);

            Eigen::Vector2d e;
            e(0) = imagePoints[i].x - proj(0);
            e(1) = imagePoints[i].y - proj(1);

            cost += e.squaredNorm();
            Eigen::Matrix<double, 2, 6> J;
            J << -fx * inv_z,
                    0,
                    fx * pc[0] * inv_z2,
                    fx * pc[0] * pc[1] * inv_z2,
                    -fx - fx * pc[0] * pc[0] * inv_z2,
                    fx * pc[1] * inv_z,
                    0,
                    -fy * inv_z,
                    fy * pc[1] * inv_z2,
                    fy + fy * pc[1] * pc[1] * inv_z2,
                    -fy * pc[0] * pc[1] * inv_z2,
                    -fy * pc[0] * inv_z;
            H += J.transpose()*J;
            b += -J.transpose()*e;
        }

        cost /= N;


        Vector6d dx = H.ldlt().solve(b);

        if(isnan(dx(0))){
            break;
        }

        if(iter> 0 && cost > lastCost){
            cout << "iters :" << iter << " cost:" << cost << endl;
            break;
        }

        // 更新权重
        T = Sophus::SE3d::exp(dx)*T;
        lastCost = cost;
        cout << "iters :" << iter << " cost:" << cost << endl;

        if(dx.norm() < 1e-6){
            break;
        }
    }

}

void p3p(){

}
/** https://blog.csdn.net/jessecw79/article/details/82945918
 * 与其他方法相比，EPnP方法的复杂度为O(n)。对于点对数量较多的PnP问题，非常高效。
    核心思想是将三维点表示为4个控制点的组合；优化也只针对4个控制点，所以速度很快；在求解 Mx = 0时，最多考虑了4个奇异向量，因此精度也很高。
 * @param imagePoints
 * @param objectPoints
 * @param K
 * @param T
 */
void epnp(vector<Point2f>& imagePoints, vector<Point3f>& objectPoints, Mat& K, Sophus::SE3d& T){
    //
    int N = imagePoints.size();

    Eigen::Matrix3d K_;
    Eigen::MatrixXd p3d(N, 3);
    Eigen::MatrixXd p2d(N, 2);

    cv2eigen(K, K_);
    for (int i=0; i<N; i++){
        p3d(i, 0) = objectPoints[i].x;
        p3d(i, 1) = objectPoints[i].y;
        p3d(i, 2) = objectPoints[i].z;

        p2d(i, 0) = imagePoints[i].x;
        p2d(i, 1) = imagePoints[i].y;
    }

    My_EPNP epnp(K_, p3d, p2d);
    Matrix3d R;
    Vector3d t;
    epnp.estimate(R, t);

    T = Sophus::SE3d(R, t);
}


void pnp_g2o(vector<Point2f>& imagePoints, vector<Point3f>& objectPoints, Mat& K, Sophus::SE3d& T){

    Eigen::Matrix3d K_;

    cv::cv2eigen(K, K_);
//    K_ <<
//            K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
//            K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
//            K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;

    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    VertexSE3Pose * vertexSe3Pose = new VertexSE3Pose();
    vertexSe3Pose->setId(0);
    vertexSe3Pose->setEstimate(Sophus::SE3d());

    optimizer.addVertex(vertexSe3Pose);

    for(int i=0; i<imagePoints.size(); ++i){
        Eigen::Vector3d pw (objectPoints[i].x, objectPoints[i].y, objectPoints[i].z);
        Eigen::Vector2d pi (imagePoints[i].x, imagePoints[i].y);
        EdgeProject* edgeProject = new EdgeProject(pw, K_);
        edgeProject->setId(i+1);
        edgeProject->setVertex(0, vertexSe3Pose);
        edgeProject->setMeasurement(pi);
        edgeProject->setInformation(Eigen::Matrix2d::Identity());

        optimizer.addEdge(edgeProject);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    T = vertexSe3Pose->estimate();

}

void pnp_ceres(vector<Point2f>& imagePoints, vector<Point3f>& objectPoints, Mat& K, Sophus::SE3d& T){
 // ceres

    double k[4];
    k[0] = K.at<double>(0, 0);
    k[1] = K.at<double>(1, 1);
    k[2] = K.at<double>(0, 2);
    k[3] = K.at<double>(1, 2);
    Vector3d r = rotationMatrix2Vector(T.rotationMatrix());
    Vector3d t = T.translation();

    double rt[6] = {
            r(0),
            r(1),
            r(2),
            t(0),
            t(1),
            t(2)
    };

    ceres::Problem problem;
    for (int i = 0; i < imagePoints.size(); ++i) {
        ceres::CostFunction* cost_function = PROJECT_COST::Create(k,imagePoints[i],objectPoints[i]);
        problem.AddResidualBlock(cost_function,
                                 nullptr /* squared loss */,
                                 rt);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    cout << endl;

    Matrix3d R_ = rotationVector2Matrix(Vector3d(rt[0], rt[1], rt[2]));
    Vector3d t_(rt[3], rt[4], rt[5]);
    T = Sophus::SE3d (R_, t_);
    //T = Sophus::SE3d::exp(rt);

}

void find_matches(Mat& img1, Mat& img2, Mat& img_depth1, Mat& K,  vector<Point2f>& imagePoints, vector<Point3f>& objectPoints){

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


    if (goodMatches.size()< 20)
        return;
    float fx, fy, cx, cy;
    fx = static_cast<float >(K.at<double>(0, 0));
    fy = static_cast<float >(K.at<double>(1, 1));
    cx = static_cast<float >(K.at<double>(0, 2));
    cy = static_cast<float >(K.at<double>(1, 2));

    for(auto & md : goodMatches){
        Point2f& pt1 = kpts1[md.queryIdx].pt;
        Point2f& pt2 = kpts2[md.trainIdx].pt;
        ushort d = img_depth1.at<ushort>(static_cast<int>(pt1.x), static_cast<int>(pt1.y));
        if (d<=0)
            continue;
        float z1_ = d / 5000.0;
        float x1_ =  (pt1.x - cx)*z1_ / fx;
        float y1_ =  (pt1.y - cy)*z1_ / fy;

        float x2_ = (pt2.x - cx) /fx;
        float y2_ = (pt2.y - cy) /fy;

        objectPoints.push_back(Point3f(x1_, y1_, z1_));
        imagePoints.push_back(pt2);
    }

}


int main(int argc, char** argv)
{
    Mat img1 = imread("/home/atway/code/slam/slambook/slambook2/ch7/1.png");
    Mat img2 = imread("/home/atway/code/slam/slambook/slambook2/ch7/2.png");
    Mat img_depth1 = imread("/home/atway/code/slam/slambook/slambook2/ch7/1_depth.png", CV_LOAD_IMAGE_UNCHANGED);

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    vector<Point2f> imagePoints;
    vector<Point3f> objectPoints;
    find_matches(img1,img2, img_depth1, K, imagePoints, objectPoints);
    cout << "match size: " << imagePoints.size() << endl;

    // DLT
    Sophus::SE3d T;
    if(true)
   {
        pnp_dlt(imagePoints, objectPoints, K, T);
        cout << "DLT T :"  << T.matrix() << endl;
        /**
         *  0.995842   -0.0901523   -0.0131136 -0.000601086
           0.0884572     0.991298   -0.0974868   0.00111193
           0.0217882    0.0959214      0.99515   0.00392255
         */
    }
    if(true)
    {
        pnp_gauss_newton(imagePoints, objectPoints,K, T);
        cout << " guass newton T :"  << T.matrix() << endl;
        /**
         *    0.998417  -0.0453521  -0.0332594 -0.00170281
              0.0448573    0.998874  -0.0154744   0.0185632
              0.0339237    0.013958    0.999327   0.0322711
         */
    }

    // opencv
    {
        Mat rvec, tvec;
        // opencv
        solvePnP(objectPoints, imagePoints, K, Mat(), rvec, tvec);

        Mat R;
        cv::Rodrigues(rvec, R);
        cout << "rvec: " << R << endl;
        cout << "tvec: " << tvec.t() << endl;
        /**
         * rvec: [0.9984172489713786, -0.04535207861145087, -0.03325937344636642;
                 0.04485732013981071, 0.9988735471532437, -0.01547441848248888;
                 0.03392370537395074, 0.01395799996838525, 0.999326951728304]
          tvec: [-0.001702807605625866, 0.01856320656120255, 0.03227109049056039]
         */
    }
    // ceres;
    if(true)
    {

        T = Sophus::SE3d(Matrix3d::Identity(), Vector3d(0., 0., 0.));
        pnp_ceres(imagePoints, objectPoints, K, T);
        cout << " ceres  T :"  << T.matrix() << endl;
        /**
         *
         *  ceres  T :   0.998417   -0.0453524   -0.0332631  -0.0016995
                         0.0448574   0.998873    -0.015481    0.0185691
                         0.0339277   0.0139644    0.999327    0.0322941
         */
    }
    if(true)
    {
        T = Sophus::SE3d(Matrix3d::Identity(), Vector3d(0., 0., 0.));
        pnp_g2o(imagePoints, objectPoints, K, T);
        cout << " g2o  T :"  << T.matrix() << endl;
    }
    // EPNP
    {
        T = Sophus::SE3d(Matrix3d::Identity(), Vector3d(0., 0., 0.));
        epnp(imagePoints, objectPoints, K, T);
        cout << " Epnp  T :"  << T.matrix() << endl;

        pnp_gauss_newton(imagePoints, objectPoints,K, T);
        cout << " guass newton T :"  << T.matrix() << endl;
    }

    return 0;
}


