//
// Created by atway on 2020/12/4.
//
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

void match(const Mat& desc1, const Mat& desc2, std::vector<DMatch>& matches){

    cv::Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create();

    std::vector<std::vector<DMatch>> pair_matches;
    matcher->knnMatch(desc1, desc2, pair_matches, 2);

    for (int i = 0; i < pair_matches.size(); ++i) {
        if(pair_matches[i].size()<2) continue;
        DMatch& m0 = pair_matches[i][0];
        DMatch& m1 = pair_matches[i][1];
        if (m0.distance < 0.6 * m1.distance){
            matches.push_back(m0);
        }
    }
}

bool calcHomography(const std::vector<KeyPoint>& kpts1, const  std::vector<KeyPoint>& kpts2,const  std::vector<DMatch>& matches, std::vector<DMatch>& goodMatches, Mat& H){

    vector<Point2f> srcpts, dstpts;

    for(auto& m: matches){
        srcpts.push_back(kpts1[m.queryIdx].pt);
        dstpts.push_back(kpts2[m.trainIdx].pt);
    }
    // OutputArray mask, int method = 0, double ransacReprojThreshold = 3
    vector<int> mask;
    H = findHomography(srcpts, dstpts,mask, cv::RANSAC, 5);

    int size =  countNonZero(mask);

    for (int i = 0; i < mask.size(); ++i) {
        if(mask[i])
            goodMatches.push_back(matches[i]);
    }

}

void warpPoints(vector<Point2f>& srcPts, vector<Point2f>& dstPts, Mat& M){
    double * h1 = M.ptr<double>(0);
    double * h2 = M.ptr<double>(1);
    double * h3 = M.ptr<double>(2);

    for(auto& pt : srcPts){

        double w = h3[0]*pt.x + h3[1]*pt.y + h3[2];
        double x = (h1[0]*pt.x + h1[1]*pt.y + h1[2])/w;
        double y = (h2[0]*pt.x + h2[1]*pt.y + h2[2])/w;

        dstPts.push_back(Point2f(x, y));
    }
}


void laplacianBlending(Mat& leftImg, Mat& rightImg, Mat& mask, OutputArray& out){


    Mat maskColor;
    cvtColor(mask, maskColor, COLOR_GRAY2BGR);

    Mat leftImgF, rightImgF;
    leftImg.convertTo(leftImgF, CV_32FC3);
    rightImg.convertTo(rightImgF, CV_32FC3);

    int maxLevel = 4;
    // 1.0 构建高斯金子塔
    vector<Mat> leftPyrGaussLayers, rightPyrGaussLayers, maskPyrGaussLayers;
    {
        buildPyramid(leftImgF, leftPyrGaussLayers, maxLevel);
        buildPyramid(rightImgF, rightPyrGaussLayers, maxLevel);
        buildPyramid(maskColor, maskPyrGaussLayers, maxLevel);
    }

    //2.0 构建拉普拉斯金字塔
    vector<Mat> leftPyrLapLayers, rightPyrLapLayers;
    {
        for (int i = 0; i < maxLevel+1; ++i) {
            if(i<maxLevel){
                Mat upleft, upright;
                pyrUp(leftPyrGaussLayers[i+1], upleft, leftPyrGaussLayers[i].size());
                leftPyrLapLayers.push_back(leftPyrGaussLayers[i] - upleft);

                pyrUp(rightPyrGaussLayers[i+1], upright, rightPyrGaussLayers[i].size());
                rightPyrLapLayers.push_back(rightPyrGaussLayers[i] - upright);
            } else{
                leftPyrLapLayers.push_back(leftPyrGaussLayers[i]);
                rightPyrLapLayers.push_back(rightPyrGaussLayers[i]);
            }
        }
    }

    
    // 3.利用掩膜作为权重，对两幅图的拉普拉斯金字塔每层的图像进行权重相乘并相加
    vector<Mat> resultLapPyr;
    {
        for (int i = 0; i < maxLevel+1; ++i) {
            Mat A = leftPyrLapLayers[i].mul(maskPyrGaussLayers[i]);
            Mat antiMask = Scalar(1.0,1.0,1.0) - maskPyrGaussLayers[i];
            Mat B = rightPyrLapLayers[i].mul(antiMask);
            Mat blendedLevel = A+B;
            resultLapPyr.push_back(blendedLevel);
        }
    }
    // 4.恢复原图
    {
        Mat currentImg=resultLapPyr[maxLevel];
        for(int l=maxLevel-1; l>=0; l--){
            Mat up;
            pyrUp(currentImg, up, resultLapPyr[l].size());
            currentImg = up + resultLapPyr[l];
        }
        currentImg.convertTo(out, CV_8UC3);
    }



}

double compensationlight(Mat& img1, Mat& img2){
     Mat gray1, gray2;
     cvtColor(img1, gray1, COLOR_BGR2GRAY);
     cvtColor(img2, gray2, COLOR_BGR2GRAY);

     double v1 = cv::sum(gray1)[0];
     double v2 = cv::sum(gray2)[0];


    return v1/v2;

}

void findseamline(Mat& img1, Mat& img2, Mat& mask){
    Mat gray1, gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);

    // Edif
    Mat sobelX1, sobelY1;
    Sobel(gray1, sobelX1, CV_32F, 1, 0);
    Sobel(gray1, sobelY1, CV_32F, 0, 1);

    Mat sobelX2, sobelY2;
    Sobel(gray2, sobelX2, CV_32F, 1, 0);
    Sobel(gray2, sobelY2, CV_32F, 0, 1);

    Mat gray1F, gray2F;
    gray1.convertTo(gray1F, CV_32F);
    gray2.convertTo(gray2F, CV_32F);

    Mat colorDiff(gray1F.size(), CV_32F);

    Mat absGray;
    cv::absdiff(gray1F, gray2F, absGray);
    Mat absSobelX, absSobelY;
    cv::absdiff(sobelX1, sobelX2, absSobelX);
    cv::absdiff(sobelY1, sobelY2, absSobelY);

    Mat geometryDiff(gray1F.size(), CV_32F);

    for (int row = 0; row < mask.rows; ++row) {
        for (int col = 0; col < mask.cols; ++col) {
            if(mask.at<uchar>(row, col)==255){

                colorDiff.at<float>(row, col) = absGray.at<float>(row, col) / max(gray1F.at<float>(row, col), gray2F.at<float>(row, col));

                float x1 = absSobelX.at<float>(row, col) /  max(sobelX1.at<float>(row, col), sobelX2.at<float>(row, col));
                float x2 = absSobelY.at<float>(row, col) /  max(sobelY1.at<float>(row, col), sobelY2.at<float>(row, col));

                geometryDiff.at<float>(row, col) = x1*x2;
            }

        }
    }

    //criterion
    Mat criterion;
    addWeighted(colorDiff, 1, geometryDiff, 1, 0, criterion);

    // 显示
    Mat diss;
    criterion.convertTo(diss, CV_8U);

    namedWindow("out", WINDOW_NORMAL);
    imshow("out", diss);
    waitKey(0);
    vector<vector<Point>> lines;
    for (int i = 0; i <mask.cols; ++i) {
        if(mask.at<uchar>(0, i) == 0) continue;
        vector<Point> line;
        int col= i, row = 0;

        while (true){
            row++;
            for (int c = col-1; c < col+1; ++c) {

            }

        }


        lines.push_back(line);
    }

}

int main()
{
    Mat img1, img2;
    img1 = imread("/home/atway/code/slam/MVGAlgorithm/slam_algorithm/data/left.jpg");
    img2 = imread("/home/atway/code/slam/MVGAlgorithm/slam_algorithm/data/right.jpg");




    Mat desc1, desc2;
    std::vector<KeyPoint> kpts1, kpts2;

    cv::Ptr<Feature2D> sift = xfeatures2d::SIFT::create();
    sift->detectAndCompute(img1, noArray(), kpts1, desc1);
    sift->detectAndCompute(img2, noArray(), kpts2, desc2);
    std::vector<DMatch> matches;
    match(desc1, desc2, matches);

    cout << "match: " << matches.size() << endl;
    Mat H;
    std::vector<DMatch> goodMatches;
    bool ok = calcHomography(kpts1, kpts2, matches,goodMatches, H);
    cout << "goodMatches: " << goodMatches.size() << endl;
    cout << "H: " << H << endl;

    Mat invH = H.inv();

    vector<Point2f> pts{
        Point2f(0, 0),
        Point2f(img2.cols, 0),
        Point2f(0, img2.rows),
        Point2f(img2.cols, img2.rows)};

    vector<Point2f> dstPts;
    warpPoints(pts, dstPts, invH);
    std::sort(dstPts.begin(), dstPts.end(), [](Point2f& m0, Point2f& m1){
       return m0.x > m1.x;
    });

    int maxCol = static_cast<int>(ceil(dstPts[0].x));



    Mat imgMask2(img2.size(), CV_8U, Scalar(1));
    Mat imgMaskResult(imgMask2.rows, maxCol,CV_8U, Scalar(0));
    cv::warpPerspective(imgMask2, imgMaskResult, H.inv(), imgMaskResult.size());
    imgMaskResult({img1.cols, 0, maxCol-img1.cols, img1.rows}) = 0;

    imwrite("imgMaskResult.jpg", imgMaskResult);

    Mat img2Warp(imgMaskResult.size(), CV_8UC3);
    cv::warpPerspective(img2, img2Warp, H.inv(), img2Warp.size());
    Mat img2WarpOverlap;
    bitwise_and(img2Warp, img2Warp, img2WarpOverlap, imgMaskResult);

    Mat img1Warp(imgMaskResult.size(), CV_8UC3);
    img1.copyTo(img1Warp({0, 0, img1.cols, img1.rows}));
    Mat img1WarpOverlap;
    bitwise_and(img1Warp, img1Warp, img1WarpOverlap, imgMaskResult);

//    imshow("outImg", img1WarpOverlap);
//    waitKey();
//    imshow("outImg", img2WarpOverlap);
//    waitKey();
    imwrite("img1WarpOverlap.jpg", img1WarpOverlap);
    imwrite("img2WarpOverlap.jpg", img2WarpOverlap);


    double k = compensationlight(img1WarpOverlap, img2WarpOverlap);
    img2Warp *= k;

    Mat mask = Mat::zeros(imgMaskResult.size(), CV_32FC1);
    mask(Range::all(), Range(0, mask.cols * 0.5)) = 1.0;


    Mat outBlend;
    laplacianBlending(img1Warp, img2Warp, mask, outBlend);

    imwrite("laplacianBlending.jpg", outBlend);

    Mat result=img2Warp.clone();
    img1.copyTo(result({0, 0, img1.cols, img1.rows}));

    //先计算平均值吧
    for (int row = 0; row < img1.rows; ++row) {

        int start=0;
        for(int col =0; col<img1.cols; ++col){
            if(imgMaskResult.at<uchar>(row, col)==1)
            {
                break;
            }
            start++;
        }

        for (int col = start; col < img1.cols; ++col) {
            //起始点的白位置， 终止点白位置 sum
            auto& v1 = outBlend.at<Vec3b>(row, col);
            auto& v2 = outBlend.at<Vec3b>(row, col);

            float w2 = (col-start) *1.0/ (img1.cols-start);
            float w1 = 1-w2;

            result.at<Vec3b>(row, col)[0] = static_cast<uchar>(floor(v1[0]*w1 + v2[0]*w2));
            result.at<Vec3b>(row, col)[1] = static_cast<uchar>(floor(v1[1]*w1 + v2[1]*w2));
            result.at<Vec3b>(row, col)[2] = static_cast<uchar>(floor(v1[2]*w1 + v2[2]*w2));

        }
    }

    imwrite("result2.jpg", result);

    //Mat outImg, outImg2;
    //drawMatches(img1, kpts1, img2, kpts2, matches, outImg, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //drawMatches(img1, kpts1, img2, kpts2, goodMatches, outImg2, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    //imwrite("out.jpg", outImg);
    //imwrite("goodout.jpg", outImg2);

    //imshow("outImg", outImg);
    //waitKey();


    //findseamline(img1Warp, img2Warp, imgMaskResult);
    return  0;
}




