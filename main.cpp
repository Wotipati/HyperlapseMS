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
    int speedUpRate;
    std::string inputVideo, inputVideoPath, outputVideoPath;
    std::cout << "video title: ";
    std::cin >> inputVideo;
//    std::cout << "speed up rate: ";
//    std::cin >> speedUpRate;
    speedUpRate = 8;
    inputVideoPath = DATA_DIRECTORY + inputVideo;
    outputVideoPath = DATA_DIRECTORY + std::string("result_") + inputVideo;
    Hyperlapse hyperlapse(inputVideoPath, outputVideoPath, speedUpRate);
    hyperlapse.generateHypelapse();
    return 0;
}
