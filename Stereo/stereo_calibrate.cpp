//
// Created by atway on 2020/12/20.
//

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <boost/format.hpp>
#include <opencv2/core/eigen.hpp>
using namespace std;
using namespace  cv;

bool readStringList( const string& filename, vector<string>& l )
{
    l.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;
    FileNode n = fs.getFirstTopLevelNode();
    if( n.type() != FileNode::SEQ )
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it )
        l.push_back("/home/atway/soft/opencv4.1/opencv-4.1.0/samples/data/"+(string)*it);
    return true;
}


bool findCorners(Mat& img, Size& patternSize, vector<Point2f> &corners,Mat& drawMat){
    bool ok = cv::findChessboardCorners(img, patternSize, corners);
    if(ok){
        cornerSubPix(img, corners, Size(11,11), Size(-1,-1),
                     TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
                                  30, 0.01));
    }

    cvtColor(img, drawMat, COLOR_GRAY2BGR);
    cv::drawChessboardCorners(drawMat, patternSize, corners, ok);
}

void StereoCalib(vector<Mat> &leftImages,vector<Mat> &rightImages, Size boardSize, float squareSize, bool displayCorners = false, bool showRectified=true)
{
    Size imageSize =leftImages[0].size();

    // extractor corners
    vector<vector<Point2f>> leftCorners, rightCorners;
    for(int i=0; i<leftImages.size(); i++){
        vector<Point2f> corners1, corners2;
        Mat leftDrawMat, rightDrawMat;
        bool leftOk = findCorners(leftImages[i], boardSize, corners1, leftDrawMat);
        bool rightOk = findCorners(rightImages[i], boardSize, corners2, rightDrawMat);
        if(displayCorners){
            Mat drawMat;
            hconcat(leftDrawMat, rightDrawMat, drawMat);
            putText(drawMat,  to_string(i), Point2f(20, 60),  FONT_HERSHEY_COMPLEX,  1.5,  Scalar(0, 0, 255));
            imshow("corners", drawMat);
            waitKey(100);
        }

        if(leftOk && rightOk){

            leftCorners.push_back(corners1);
            rightCorners.push_back(corners2);
        }
    }
    if(displayCorners)
        destroyWindow("corners");


    // 计算世界坐标点
    vector<vector<Point3f>> objectCorners;
    for(int i=0; i<leftCorners.size(); ++i){
        vector<Point3f> corners;
        for (int h = 0; h < boardSize.height; ++h) {
            for(int w = 0; w <boardSize.width; ++w){
                corners.push_back(Point3f(w*squareSize, h*squareSize, 0.));
            }
        }
        objectCorners.push_back(corners);
    }

    //标定left camera
    Mat  leftCameraMatrix,  leftDistCoeffs;
    {
        vector<Mat> rvecs,tvecs;
        double  error = cv::calibrateCamera(objectCorners, leftCorners,  imageSize, leftCameraMatrix, leftDistCoeffs, rvecs, tvecs);
        std::cout << "left project error " << error << std::endl;

        if(showRectified){
            for(int i=0; i<leftImages.size(); i++){
                cv::Mat undist;
                cv::undistort(leftImages[i], undist, leftCameraMatrix, leftDistCoeffs, leftCameraMatrix);
                imshow("rectified", undist);
                waitKey(100);
            }
        }
    }
    Mat  rightCameraMatrix,  rightDistCoeffs;
    {
        vector<Mat> rvecs,tvecs;
        double  error = cv::calibrateCamera(objectCorners, rightCorners,  imageSize, rightCameraMatrix, rightDistCoeffs, rvecs, tvecs);
        std::cout << "right project error " << error << std::endl;
        if(showRectified){
            for(int i=0; i<rightImages.size(); i++){
                cv::Mat undist;
                cv::undistort(rightImages[i], undist, rightCameraMatrix, rightDistCoeffs, rightCameraMatrix);
                imshow("rectified", undist);
                waitKey(100);
            }
        }
    }

    std::cout << "left matrix:" << leftCameraMatrix << std::endl;
    std::cout << "left dist:" << leftDistCoeffs << std::endl;
    std::cout << "right matrix:" << rightCameraMatrix << std::endl;
    std::cout << "right dist:" << rightDistCoeffs << std::endl;
    Mat R, T, E, F;
    {
        // 双目标定
//        double error = cv::stereoCalibrate(leftObjectCorners, leftCorners, rightCorners,
//                leftCameraMatrix, leftDistCoeffs,
//                rightCameraMatrix, rightDistCoeffs,imageSize,
//                R, T, E, F, CALIB_FIX_INTRINSIC);
        //CALIB_USE_INTRINSIC_GUESS
        double rms = stereoCalibrate(objectCorners, leftCorners, rightCorners,
                                     leftCameraMatrix, leftDistCoeffs,
                                     rightCameraMatrix, rightDistCoeffs,
                                     imageSize, R, T, E, F,
                                     CALIB_FIX_ASPECT_RATIO +
                                     CALIB_ZERO_TANGENT_DIST +
                                     CALIB_USE_INTRINSIC_GUESS +
                                     CALIB_SAME_FOCAL_LENGTH +
                                     CALIB_RATIONAL_MODEL+
                                     CALIB_FIX_K3+CALIB_FIX_K4+CALIB_FIX_K5,
                                     TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-5) );

        std::cout << "stereo calibrate project error " << rms << std::endl;

        std::cout << "left matrix:" << leftCameraMatrix << std::endl;
        std::cout << "left dist:" << leftDistCoeffs << std::endl;
        std::cout << "right matrix:" << rightCameraMatrix << std::endl;
        std::cout << "right dist:" << rightDistCoeffs << std::endl;

    }


    std::cout << "R:" << R << std::endl;
    std::cout <<"T" << T.t() << std::endl;

    Mat R1,  R2,  P1, P2, Q;
    Rect  roi1, roi2;
    {
        cv::stereoRectify(leftCameraMatrix, leftDistCoeffs, rightCameraMatrix, rightDistCoeffs, imageSize,
                          R, T, R1, R2, P1, P2, Q,
                          CALIB_ZERO_DISPARITY, 1, imageSize, &roi1, &roi2);

        std::cout << "R left:" << R1 << std::endl;
        std::cout << "R right:" << R2 << std::endl;
        std::cout << "P left" << P1 << std::endl;
        std::cout << "P right:" << P2 << std::endl;
        std::cout << "Q:" <<  Q << std::endl;


    }

    Mat map11, map12, map21, map22;
    {
        initUndistortRectifyMap(leftCameraMatrix, leftDistCoeffs, R1, P1, imageSize, CV_16SC2, map11, map12);
        initUndistortRectifyMap(rightCameraMatrix, rightDistCoeffs, R2, P2, imageSize, CV_16SC2, map21, map22);
    }

    if(showRectified)
    {
        for( int i = 0; i < leftImages.size(); i++ )
        {
            Mat colorLeftImg, colorRightImg;
            cvtColor(leftImages[i], colorLeftImg, COLOR_GRAY2BGR);
            cvtColor(rightImages[i], colorRightImg, COLOR_GRAY2BGR);

            Mat imgr1, imgr2;
            remap(colorLeftImg, imgr1, map11, map12, INTER_LINEAR);
            remap(colorRightImg, imgr2, map21, map22, INTER_LINEAR);

            rectangle(imgr1, roi1, Scalar(0,0,255), 3, 8);
            rectangle(imgr2, roi2,  Scalar(0,0,255), 3, 8);

            Mat imgcat;
            hconcat(imgr1, imgr2, imgcat);
            for(int j = 0; j < imgcat.rows; j += 16 )
                line(imgcat, Point(0, j), Point(imgcat.cols, j), Scalar(0, 255, 0), 1, 8);
            imshow("rectified", imgcat);
            waitKey();
        }

    }

}

int main(){

    vector<Mat> leftImages, rightImages;
    Size  boardSize;
    float squareSize;
    bool myntData=true ;
    // read images list;
    if(myntData)
    {   squareSize= 25.f;
        boardSize= Size (9, 7);
        boost::format fmt("/home/atway/code/slam/MVGAlgorithm/slam_algorithm/data/stereo/%s%02d.jpg");
        int total=21;
        for(int i=0; i<21; i++){
            string leftfile = (fmt%"left"%i).str();
            string rightfile = (fmt%"right"%i).str();
            Mat left =imread(leftfile, IMREAD_GRAYSCALE);
            Mat right = imread(rightfile, IMREAD_GRAYSCALE);
            if(!left.empty() && !right.empty())
            {
                leftImages.push_back(left);
                rightImages.push_back(right);
            }
        }

    }
    else
    {
        string imagelistfn = "/home/atway/soft/opencv4.1/opencv-4.1.0/samples/data/stereo_calib.xml";
        squareSize= 25.f;
        boardSize= Size (9, 6);
        vector<string> imagelist;
        bool ok = readStringList(imagelistfn, imagelist);
        for(int i=0; i<imagelist.size()/2; i++){
            leftImages.push_back(imread(imagelist[i*2], IMREAD_GRAYSCALE));
            rightImages.push_back(imread(imagelist[i*2+1], IMREAD_GRAYSCALE));
        }
    }

    StereoCalib(leftImages, rightImages, boardSize, squareSize,true, true);



    /*
     *
     *Intrinsics left: {equidistant, width: 752, height: 480,
     * k2: -0.01509692738043893, k3: -0.03488162590077018, k4: 0.05372013323333781, k5: -0.02979241878641033,
     * mu: 369.61759181873406988, mv: 369.54921986988949811,
     * u0: 374.59465658384857534, v0: 235.16259026829271761}
    Intrinsics right: {equidistant, width: 752, height: 480,
     k2: -0.02267636068078869, k3: 0.00012273207057233, k4: -0.00412066517603011, k5: 0.00077768928909407,
     mu: 369.65416692810941868, mv: 369.56670990194794513,
     u0: 372.31377002663646181, v0: 230.59248921651672504}

    Extrinsics right to left: {
     rotation: [
     0.99998366413954298, -0.00328144799649875, 0.00468012319281784,
     0.00328505774061074, 0.99999431247580406, -0.00076381390718632,
     -0.00467759015888851, 0.00077917590455050, 0.99998875645439900],
     translation:
     [-120.19933355765978433, -0.06117490766245021, 0.73604873985802033]}
     *
     */
    return 0;
}

