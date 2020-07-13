//
// Created by atway on 2020/7/7.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

using namespace std;
// 创建数据
void createData(double* weights, std::vector<double>& xData, std::vector<double>& yData){
    int N=100;
    double sigma = 1.0;
    cv::RNG rng;

    for(int i=0; i<N; i++){
        double x = rng.uniform(0.0, 1.0);
        double y = exp(weights[0]*x*x + weights[1]*x + weights[2]) + rng.gaussian(sigma);
        xData.push_back(x);
        yData.push_back(y);
    }
}


// 矩阵计算
void train(std::vector<double>& xData, std::vector<double>& yData, double* weights, int iters){

    int N = xData.size();

    double lastCost = 0;
    for (int iter=0; iter<iters; ++iter){

        // 定义雅克比矩阵和H矩阵
        // 每行对权重的导数 Nx3
        Eigen::MatrixXd J (N, 3);
        Eigen::Matrix3d H;
        Eigen::VectorXd errors(N);
        double cost=0.0;
        for(int i =0; i<N; ++i){
            double x = xData[i];
            double y = yData[i];
            double predict = exp(weights[0]*x*x + weights[1]*x + weights[2]);
            double error = y  - predict;

            J(i, 0) = -x*x*predict;
            J(i, 1) = -x*predict;
            J(i, 2)= -predict;

            errors [i] = error;
            cost += error*error;
        }

        H = J.transpose()*J;
        Eigen::Vector3d  b = -J.transpose() * errors;

        // 求解矩阵
        Eigen::Vector3d dx = H.ldlt().solve(b);

        if(isnan(dx[0])){
           cout << "error nan" << endl;
            break;;
        }

        if(iter > 0 && cost > lastCost){
            break;
        }

        // 更新权重
        weights[0] += dx[0];
        weights[1] += dx[1];
        weights[2] += dx[2];

        lastCost = cost;

        cout << "iters: " << iter << "  cost: " << cost / N << endl;
    }
}

// 求和计算
void train_(std::vector<double>& xData, std::vector<double>& yData, double* weights, int iters){

    int N = xData.size();

    double lastCost = 0;
    for (int iter=0; iter<iters; ++iter){

        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        Eigen::Vector3d b = Eigen::Vector3d ::Zero();
        double cost=0.0;
        for(int i =0; i<N; ++i){
            double x = xData[i];
            double y = yData[i];
            double predict = exp(weights[0]*x*x + weights[1]*x + weights[2]);
            double error = y  - predict;

            Eigen::Vector3d J;
            J(0) = -x*x*predict;
            J(1) = -x*predict;
            J(2)= -predict;

            H += J*J.transpose();
            b += -J * error;

            cost += error*error;
        }

        // 求解矩阵
        Eigen::Vector3d dx = H.ldlt().solve(b);

        if(isnan(dx[0])){
            cout << "error nan" << endl;
            break;;
        }

        if(iter > 0 && cost > lastCost){
            break;
        }

        // 更新权重
        weights[0] += dx[0];
        weights[1] += dx[1];
        weights[2] += dx[2];

        lastCost = cost;

        cout << "iters: " << iter << "  cost: " << cost / N << endl;
    }
}



int main(int argc, char** argv){

    std::vector<double> xData, yData;
    double real_weights[3] = {1.0, 2.0, 1.0};
    createData(real_weights, xData, yData);

    double weights[3] = {2., -1., 5.};
    train_(xData, yData, weights, 1000);

    for (auto& v : weights)
        cout << v << " ";
    cout << endl;
    return 0;
}