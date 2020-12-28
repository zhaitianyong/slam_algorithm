//
// Created by atway on 2020/12/12.
//
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include  <iostream>

using namespace  cv;
using namespace  std;



int main(){


    VideoCapture capture;
    bool ok = capture.open(0, CAP_ANY);
    if(!ok){
        cout << "open camera failed" << endl;
        return -1;
    }
    Mat frame, gray;
    int currFocus=0;
    double stdVal=0.0;
    while(ok){
        capture.set(CAP_PROP_FOCUS, currFocus);
        capture.read(frame);
        if(!frame.empty()){
            Mat edge; Scalar mean, stddev;
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            cv::Laplacian(gray, edge, CV_64F);
            cv::meanStdDev(edge, mean,stddev);
            cout << "mean:"<< mean  << " stddev" << stddev<< endl;
            imshow("out", frame);
            waitKey();
            currFocus+=10;

        }

    }


    return 0;
}
