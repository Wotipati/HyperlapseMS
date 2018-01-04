//
//  main.cpp
//  HyperlapseMS
//
//  Created by Seita Kayukawa on 2017/12/31.
//  Copyright © 2017年 Seita Kayukawa. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>

#include "common.h"
#include "Hyperlapse.h"

int main(int argc, const char * argv[]) {
    std::string inputVideo, inputVideoPath, outputVideoPath;
    //    std::cout << "video title: " << std::endl;
    //    std::cin >> inputVideo;
    inputVideo = "video.mp4";
    //inputVideo = "test.mp4";
    inputVideoPath = DATA_DIRECTORY + inputVideo;
    outputVideoPath = DATA_DIRECTORY + std::string("result_") + inputVideo;
    Hyperlapse hyperlapse(inputVideoPath, outputVideoPath);
    hyperlapse.generateHypelapse();
    return 0;
}
