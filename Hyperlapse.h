//
//  Hyperlapse.h
//  HyperlapseMS
//
//  Created by Seita Kayukawa on 2017/12/31.
//  Copyright © 2017年 Seita Kayukawa. All rights reserved.
//

#ifndef Hyperlapse_h
#define Hyperlapse_h

#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>

#include "common.h"

class Hyperlapse{
public:
    Hyperlapse(std::string inputVideoPath, std::string outputVideoPath);
    void generateHypelapse();
    
private:
    void calcFrameMatchingCost();
    float costFunction(std::vector<cv::DMatch>& matches,
                       std::vector<cv::KeyPoint>& keyPointsI,
                       std::vector<cv::KeyPoint>& keyPointsJ);
    float reprojectionError(std::vector<cv::Point2f>& keyPointsI,
                            std::vector<cv::Point2f>& keyPointsJ,
                            cv::Mat& homography);
    
    
    void pathSelection();
    cv::Mat initialization(cv::Mat Cm);
    void populateDv(cv::Mat Cm, cv::Mat& Dv, cv::Mat& Tv);
    void traceBackMinCostPath(cv::Mat& Dv, cv::Mat& Tv,
                              std::vector<cv::Mat>& inputFrames,
                              std::vector<cv::Mat>& optimalFrames);
    
    void stabilizeFrames();
    
    void writeHyperlapse();
    void writeSimpleHyperlapse();
    
    std::string inputVideoPath_;
    std::string outputVideoPath_;
    std::vector<cv::Mat> inputFrames_;
    cv::Mat Cm_;
    std::vector<cv::Mat> optimalFrames_;
    cv::Ptr<cv::videostab::IFrameSource> stabilizedFrames_;
    
    int frameWidth_, frameHeight_;
    int diagonal_;
    int totalFrameCount_;
    
    float gamma_;
    
    float tauC_;
    float tauS_ = 200;
    float tauA_ = 200;
    
    int window_ = 32;
    int gap_ = 4;
    
    int speedUpRate_ = 8;
    int lambdaS_ = 200;
    int lambdaA_ = 80;
    
    
};

class VecSource : public cv::videostab::IFrameSource {
public:
    VecSource(std::vector<cv::Mat>* frames) {
        this->frames = frames;
        reset();
    }
    
    virtual void reset() {
        index = 0;
    }
    
    virtual cv::Mat nextFrame() {
        if (index == frames->size()) {
            cv::Mat m;
            return m;
        }
        
        cv::Mat frame = (*frames)[index];
        index++;
        return frame;
    }
    
private:
    std::vector<cv::Mat>* frames;
    int index;
};

#endif /* Hyperlapse_h */
