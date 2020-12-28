// Copyright 2018 Slightech Co., Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include <boost/format.hpp>
#include "mynteye/api/api.h"

MYNTEYE_USE_NAMESPACE

int main(int argc, char const *argv[]) {
  auto &&api = API::Create(0, nullptr);
  if (!api) return 1;

  bool ok;
  auto &&request = api->SelectStreamRequest(&ok);
  if (!ok) return 1;
  api->ConfigStreamRequest(request);
  api->Start(Source::VIDEO_STREAMING);

  double fps;
  double t = 0.01;
    std::cout << "fps:" << std::endl;

    std::cout << "Intrinsics left: {" << *api->GetIntrinsicsBase(Stream::LEFT)
              << "}"<< std::endl;
    std::cout << "Intrinsics right: {" << *api->GetIntrinsicsBase(Stream::RIGHT)
              << "}"<< std::endl;
    std::cout << "Extrinsics right to left: {"
              << api->GetExtrinsics(Stream::RIGHT, Stream::LEFT) << "}"<< std::endl;

  // 获得相机IMU标定参数
    std::cout<< "Motion intrinsics: {" << api->GetMotionIntrinsics() << "}"<< std::endl;
    std::cout << "Motion extrinsics left to imu: {"
              << api->GetMotionExtrinsics(Stream::LEFT) << "}"<< std::endl;

  cv::namedWindow("frame");

  int ct = 0;
  boost::format fmt("/home/atway/code/slam/MVGAlgorithm/slam_algorithm/data/stereo/%s%02d.jpg");
  while (true) {
    api->WaitForStreams();

    auto &&left_data = api->GetStreamData(Stream::LEFT);
    auto &&right_data = api->GetStreamData(Stream::RIGHT);

    cv::Mat img;
    if (!left_data.frame.empty() && !right_data.frame.empty()) {
      //double t_c = cv::getTickCount() / cv::getTickFrequency();
      //fps = 1.0/(t_c - t);
      //printf("\b\b\b\b\b\b\b\b\b%.2f", fps);
      //t = t_c;
      cv::hconcat(left_data.frame, right_data.frame, img);
      cv::imshow("frame", img);
    }

    char key = static_cast<char>(cv::waitKey(1));

    if(key == 32){

        if(!left_data.frame.empty() && !right_data.frame.empty()){
            std::string leftname = (fmt%"left" %ct).str();
            std::string rightname = (fmt%"right"%ct).str();
            std::cout<< leftname << std::endl;
            cv::imwrite(leftname, left_data.frame);
            cv::imwrite(rightname, right_data.frame);
            ct+=1;
        }
    }

    if (key==27 || key == 'q' || key == 'Q') {  // ESC/Q
      break;
    }

  }

  api->Stop(Source::VIDEO_STREAMING);
  return 0;
}
