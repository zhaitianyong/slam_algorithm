//
// Created by atway on 2020/7/13.
//

#include "My_EPNP.h"

My_EPNP::My_EPNP(Eigen::Matrix3d &K, Eigen::MatrixXd& p3d, Eigen::MatrixXd& p2d) {
    this->K = K;
    this->vp3d = p3d;
    this->vp2d = p2d;

    fx = K(0, 0);
    fy = K(1, 1);
    cx = K(0, 2);
    cy = K(1, 2);
}

void  My_EPNP::estimate(Matrix3d &R, Vector3d &t) {
    // 计算世界坐标的4个控制点
    std::vector<Eigen::Vector3d> control_pts_world;
    choose_control_points(control_pts_world);

    // 根据控制点，计算每个点的系数,也就是 ai1 ai2 ai3 ai4
    vector<Vector4d> hb;
    compute_barycentric_coordinates(control_pts_world, hb);

    // 计算姿态
    camera_position_from_control_points(hb, control_pts_world, R, t);
}

void My_EPNP::choose_control_points(std::vector<Eigen::Vector3d>& control_pts_world) {

    Eigen::Vector3d c1;
    c1(0) = vp3d.col(0).mean();
    c1(1) = vp3d.col(1).mean();
    c1(2) = vp3d.col(2).mean();

    int N = vp3d.rows();

    Eigen::MatrixXd A(N, 3);

    for (int i=0; i<N; ++i){
        A.row(i) = vp3d.row(i) - c1.transpose();
    }
    // 3x3
    Eigen::Matrix3d H = A.transpose()*A;

    Eigen::JacobiSVD<Eigen::MatrixXd> solver_H(H, Eigen::ComputeFullU|Eigen::ComputeFullV);
    Eigen::Vector3d  sigma_H = solver_H.singularValues();
    Eigen::Matrix3d  V = solver_H.matrixV();

    Eigen::Vector3d c2 = c1 + sqrt(sigma_H(0))*V.col(0);
    Eigen::Vector3d c3 = c1 + sqrt(sigma_H(1))*V.col(1);
    Eigen::Vector3d c4 = c1 + sqrt(sigma_H(2))*V.col(2);

    control_pts_world.push_back(c1);
    control_pts_world.push_back(c2);
    control_pts_world.push_back(c3);
    control_pts_world.push_back(c4);
}

void My_EPNP::compute_barycentric_coordinates(vector<Vector3d>& control_pts_world, vector<Vector4d>& hb){

    Matrix4d C;

    for(int i=0; i<control_pts_world.size(); ++i){
        C(0, i) = control_pts_world[i](0);
        C(1, i) = control_pts_world[i](1);
        C(2, i) = control_pts_world[i](2);
        C(3, i) = 1.;
    }
    Matrix4d C_inv;
    C_inv = C.inverse();

    int N = vp3d.rows();


    for (int i=0; i<N; ++i){
        Eigen::Vector4d ptw ( 0.0, 0.0, 0.0, 1.0 );
        ptw.block ( 0, 0, 3, 1 ) = vp3d.row(i);
        hb.push_back(C_inv * ptw);
    }

}

void My_EPNP::camera_position_from_control_points(vector<Vector4d>& hb, vector<Vector3d>& control_pts_world, Matrix3d& R, Vector3d& t) {

    // 构造M矩阵 2Nx12
    // init M
    MatrixXd M;
    const int n = vp3d.rows();
    {
        M.resize ( 2*n, 12 );

        for ( int i = 0; i < n; i ++ ) {
            // get alphas
            const Eigen::Vector4d &alphas = hb.at(i);

            const double &alpha_i1 = alphas(0);
            const double &alpha_i2 = alphas(1);
            const double &alpha_i3 = alphas(2);
            const double &alpha_i4 = alphas(3);

            // get uv
            const double &u = vp2d(i, 0);
            const double &v = vp2d(i, 1);

            // the first line
            M(2 * i, 0) = alpha_i1 * fx;
            M(2 * i, 1) = 0.0;
            M(2 * i, 2) = alpha_i1 * (cx - u);

            M(2 * i, 3) = alpha_i2 * fx;
            M(2 * i, 4) = 0.0;
            M(2 * i, 5) = alpha_i2 * (cx - u);

            M(2 * i, 6) = alpha_i3 * fx;
            M(2 * i, 7) = 0.0;
            M(2 * i, 8) = alpha_i3 * (cx - u);

            M(2 * i, 9) = alpha_i4 * fx;
            M(2 * i, 10) = 0.0;
            M(2 * i, 11) = alpha_i4 * (cx - u);


            // for the second line
            M(2 * i + 1, 0) = 0.0;
            M(2 * i + 1, 1) = alpha_i1 * fy;
            M(2 * i + 1, 2) = alpha_i1 * (cy - v);

            M(2 * i + 1, 3) = 0.0;
            M(2 * i + 1, 4) = alpha_i2 * fy;
            M(2 * i + 1, 5) = alpha_i2 * (cy - v);

            M(2 * i + 1, 6) = 0.0;
            M(2 * i + 1, 7) = alpha_i3 * fy;
            M(2 * i + 1, 8) = alpha_i3 * (cy - v);

            M(2 * i + 1, 9) = 0.0;
            M(2 * i + 1, 10) = alpha_i4 * fy;
            M(2 * i + 1, 11) = alpha_i4 * (cy - v);
        }

    }

    // 根据论文只要取前4个特征向量就可以
    MatrixXd eigen_vectors;
    {
        Eigen::Matrix<double, 12, 12> MTM = M.transpose() * M;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 12, 12>> es ( MTM );
        MatrixXd vectors_ = es.eigenvectors();
        eigen_vectors = vectors_.leftCols(4);
    }

    // 求解,
    Eigen::MatrixXd L6_10 = Eigen::MatrixXd::Zero(6, 10);
    Eigen::VectorXd rho(6, 1);
    {
        computeL6_10(eigen_vectors, L6_10);
        computeRho(rho, control_pts_world);
    }


    vector<Matrix3d> vR;
    vector<Vector3d> vt;
    vector<double> errors;
    // N = 2
    {

        Eigen::Matrix3d tempR;
        Eigen::Vector3d tempT;
        Vector4d betas;
        find_betas_approx_2(L6_10, rho, betas);
        gauss_newton(L6_10, rho, betas);
        MatrixXd vp3d_camera(n, 3);
        calcCameraPoints(betas, eigen_vectors, hb, vp3d_camera);
        computeRt(vp3d_camera, tempR, tempT);
        double error = reprojectError(tempR, tempT);
        vR.push_back(tempR);
        vt.push_back(tempT);
        errors.push_back(error);
    }
    // N = 3
    {

        Eigen::Matrix3d tempR;
        Eigen::Vector3d tempT;
        Vector4d betas;
        find_betas_approx_3(L6_10, rho, betas);
        gauss_newton(L6_10, rho, betas);
        MatrixXd vp3d_camera(n, 3);
        calcCameraPoints(betas, eigen_vectors, hb, vp3d_camera);
        computeRt(vp3d_camera, tempR, tempT);
        double error = reprojectError(tempR, tempT);
        vR.push_back(tempR);
        vt.push_back(tempT);
        errors.push_back(error);
    }
    // N = 4
    {

        Eigen::Matrix3d tempR;
        Eigen::Vector3d tempT;
        Vector4d betas;
        find_betas_approx_4(L6_10, rho, betas);
        gauss_newton(L6_10, rho, betas);
        MatrixXd vp3d_camera(n, 3);
        calcCameraPoints(betas, eigen_vectors, hb, vp3d_camera);
        computeRt(vp3d_camera, tempR, tempT);
        double error = reprojectError(tempR, tempT);
        vR.push_back(tempR);
        vt.push_back(tempT);
        errors.push_back(error);
    }

    int minIndex=0;
    double minError = errors[0];
    for(int i=1; i<errors.size(); i++){
        if(minError < errors[i] ){
            minError = errors[i];
            minIndex = i;
        }
    }

    R = vR.at(minIndex);
    t = vt.at(minIndex);

}

double My_EPNP::calcCameraPoints(Vector4d& betas,  MatrixXd& eigen_vectors, vector<Vector4d>& hb, MatrixXd& vp3d_camera) {
    //计算 control_pts_camera;
    int n = hb.size();

    Matrix<double, 12, 1> x_v = Matrix<double, 12, 1>::Zero();
    for(int i=0; i<4; ++i){
        x_v += betas(i)*eigen_vectors.col(i);
    }
    vector<Vector3d> control_pts_camera;
    control_pts_camera.push_back(x_v.block(0, 0, 3, 1));
    control_pts_camera.push_back(x_v.block(3, 0, 3, 1));
    control_pts_camera.push_back(x_v.block(6, 0, 3, 1));
    control_pts_camera.push_back(x_v.block(9, 0, 3, 1));
    //根据控制点和hb ->a1, a2, a3,a4

    for(int i=0; i<n; ++i){
        vp3d_camera.block(i, 0, 1, 3) =  (hb[i](0)*control_pts_camera[0]
                                          +hb[i](1)*control_pts_camera[1]
                                          +hb[i](2)*control_pts_camera[2]
                                          +hb[i](3)*control_pts_camera[3]).transpose();
    }

}


double My_EPNP::reprojectError(Matrix3d& R, Vector3d& t){

    int n = vp3d.size();

    double error=0.0;
    for (int i=0; i < n; ++i){
       Vector3d pc =  K * (R * (vp3d.row(i).transpose()) + t);
       pc /= pc(2);

       Vector2d diff = vp2d.row(i).transpose() - pc.head(2);

       error += diff.dot(diff);

    }
    error /= n;
    return error;
}

void My_EPNP::computeRho(Eigen::VectorXd& rho, vector<Vector3d>& p3d_w){
    Eigen::Vector3d control_point_a, control_point_b, control_point_diff;
    double diff_pattern[6][6] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

    for (int i = 0; i < 6; i++){
      Vector3d  v01 = p3d_w.at(diff_pattern[i][0]) - p3d_w.at(diff_pattern[i][1]);
      rho(i, 0) = v01.dot(v01);
    }
}

void My_EPNP::computeL6_10(const Eigen::MatrixXd& U, Eigen::MatrixXd& L6_10){
    Eigen::MatrixXd V = U.block(0, 0, 12, 4);
    Eigen::MatrixXd DiffMat = Eigen::MatrixXd::Zero(18, 4);
    double diff_pattern[6][6] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

    for (int i = 0; i < 6; i++){
        DiffMat.block(3*i, 0, 3, 4) = V.block(3 * diff_pattern[i][0], 0, 3, 4) - V.block(3 * diff_pattern[i][1], 0, 3, 4);
    }

    Eigen::Vector3d v1, v2, v3, v4;
    for (int i = 0; i < 6; i++){
        v1 = DiffMat.block(3*i, 0, 3, 1);
        v2 = DiffMat.block(3*i, 1, 3, 1);
        v3 = DiffMat.block(3*i, 2, 3, 1);
        v4 = DiffMat.block(3*i, 3, 3, 1);

        L6_10.block(i, 0, 1, 10) << v1.dot(v1), 2*v1.dot(v2), v2.dot(v2), 2*v1.dot(v3), 2*v2.dot(v3),
                v3.dot(v3), 2*v1.dot(v4), 2*v2.dot(v4), 2*v3.dot(v4), v4.dot(v4);
    }
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_1 = [B11 B12     B13         B14]
// N = 4
void My_EPNP::find_betas_approx_4(Eigen::MatrixXd L6_10, Eigen::VectorXd& rho,
                                  Vector4d&  betas)
{
    MatrixXd L_approx(6, 4);
    L_approx.block(0, 0, 6, 2) = L6_10.block(0,0, 6,2);
    L_approx.block(0, 2, 6, 1) = L6_10.block(0,3, 6,1);
    L_approx.block(0, 3, 6, 1) = L6_10.block(0,6, 6,1);


    Vector4d  b4 = L_approx.fullPivHouseholderQr().solve(rho);


    if (b4(0) < 0) {
        betas(0) = sqrt(-b4(0));
        betas(1) = -b4(1) / betas(0);
        betas(2) = -b4(2) / betas(0);
        betas(3) = -b4(3) / betas(0);
    } else {
        betas(0) = sqrt(b4(0));
        betas(1) = b4(1) / betas(0);
        betas(2) = b4(2) / betas(0);
        betas(3) = b4(3) / betas(0);
    }
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_2 = [B11 B12 B22                            ]
// N = 2
void My_EPNP::find_betas_approx_2(Eigen::MatrixXd L6_10, Eigen::VectorXd& rho,
                                  Vector4d&  betas)
{
    MatrixXd L_approx(6, 3);
    L_approx = L6_10.block(0,0, 6,3);


    Vector3d  b3 = L_approx.fullPivHouseholderQr().solve(rho);


    if (b3(0) < 0) {
        betas(0) = sqrt(-b3(0));
        betas(1) = (b3(2) < 0)? sqrt(-b3(2)):0.0;
    } else {
        betas(0) = sqrt(b3(0));
        betas(1) = (b3(2) > 0)? sqrt(b3(2)):0.0;
    }
    betas(2) = 0.;
    betas(3) = 0.;
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_3 = [B11 B12 B22 B13 B23                    ]
// N = 3
void My_EPNP::find_betas_approx_3(Eigen::MatrixXd L6_10, Eigen::VectorXd& rho,
                                  Vector4d&  betas)
{
    MatrixXd L_approx(6, 5);
    L_approx = L6_10.block(0,0, 6,5);

    Matrix<double, 5, 1>  b5 = L_approx.fullPivHouseholderQr().solve(rho);

    if (b5(0) < 0) {
        betas(0) = sqrt(-b5(0));
        betas(1) = (b5(2) < 0)? sqrt(-b5(2)):0.0;
    } else {
        betas(0) = sqrt(b5(0));
        betas(1) = (b5(2) > 0)? sqrt(b5(2)):0.0;
    }
    if(b5(1) < 0){
        betas(0) = - betas(0);
    }
    betas(2) = b5(3) / betas(0);
    betas(3) = 0.;
}

// 高斯牛顿优化
void  My_EPNP::gauss_newton(Eigen::MatrixXd L6_10, Eigen::VectorXd& rho,Vector4d& betas){

    const int iter_num = 5;

    for ( int nit = 0; nit < iter_num; nit ++ ) {

        // construct J
        Eigen::Matrix<double, 6, 4> J;
        for ( int i = 0; i < 6; i ++ ) {
            // [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
            J ( i, 0 ) = 2 * betas[0] * L6_10 ( i, 0 ) + betas[1]* L6_10 ( i, 1 ) + betas[2]*L6_10 ( i, 3 ) + betas[3]*L6_10 ( i, 6 );
            J ( i, 1 ) = betas[0] * L6_10 ( i, 1 ) + 2 * betas[1]* L6_10 ( i, 2 ) + betas[2]*L6_10 ( i, 3 ) + betas[3]*L6_10 ( i, 7 );
            J ( i, 2 ) = betas[0] * L6_10 ( i, 3 ) + betas[1]* L6_10 ( i, 4 ) + 2 * betas[2]*L6_10 ( i, 5 ) + betas[3]*L6_10 ( i, 8 );
            J ( i, 3 ) = betas[0] * L6_10 ( i, 6 ) + betas[1]* L6_10 ( i, 7 ) + betas[2]*L6_10 ( i, 8 ) + 2 * betas[3]*L6_10 ( i, 9 );
        }

        Eigen::Matrix<double, 4, 6> J_T = J.transpose();
        Eigen::Matrix<double, 4, 4> H = J_T * J;

        // Compute residual
        // [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
        // [B00 B01 B11 B02 B12 B22 B03 B13 B23 B33]
        Eigen::Matrix<double, 10, 1> bs;
        bs << betas[0]*betas[0], betas[0]*betas[1], betas[1]*betas[1], betas[0]*betas[2], betas[1]*betas[2],
                betas[2]*betas[2], betas[0]*betas[3], betas[1]*betas[3], betas[2]*betas[3], betas[3]*betas[3];
        Eigen::Matrix<double, 6, 1> residual = L6_10 * bs - rho;

        //std::cout << "Error " << residual.transpose() * residual << "\n";

        // Solve J^T * J \delta_beta = -J^T * residual;
        Eigen::Matrix<double, 4, 1> delta_betas = H.fullPivHouseholderQr().solve ( -J_T * residual );

        // update betas;
        betas += delta_betas;
    } //iter n times.

}

// svd  icp 计算方式
void  My_EPNP::computeRt(MatrixXd& p3d_c, Matrix3d& R, Vector3d& t){

    Vector3d pw_center = Vector3d ::Zero();
    Vector3d pc_center = Vector3d::Zero();
    int N = p3d_c.size();
    pw_center(0) = vp3d.col(0).mean();
    pw_center(0) = vp3d.col(1).mean();
    pw_center(0) = vp3d.col(2).mean();

    pc_center(0) = p3d_c.col(0).mean();
    pc_center(0) = p3d_c.col(1).mean();
    pc_center(0) = p3d_c.col(2).mean();


    MatrixXd Pw(N, 3);
    MatrixXd Pc(N, 3);

    Pw = (vp3d.transpose()-pw_center).transpose();

    Pc = (p3d_c.transpose()-pc_center).transpose();

    Matrix3d W = Pc.transpose()*Pw; //3x3

    Eigen::JacobiSVD<MatrixXd> solver(W, Eigen::ComputeFullU|Eigen::ComputeThinV);
    Matrix3d U = solver.matrixU();
    Matrix3d V = solver.matrixV();

    R = U*V.transpose();
    if(R.determinant() < 0){
        R = -R;
    }

    t = pc_center - R*pw_center;
}