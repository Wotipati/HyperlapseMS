//
//  Hyperlapse.cpp
//  HyperlapseMS
//
//  Created by Seita Kayukawa on 2017/12/31.
//  Copyright © 2017年 Seita Kayukawa. All rights reserved.
//

#include "Hyperlapse.h"


Hyperlapse::Hyperlapse(std::string inputVideoPath, std::string outputVideoPath){
    inputVideoPath_ = inputVideoPath;
    outputVideoPath_ = outputVideoPath;
    
    cv::VideoCapture inputVideo_(inputVideoPath_);
    if (!inputVideo_.isOpened()){
        std::cout << "ERROR: Failed to open movie" << std::endl;
        exit(0);
    }
    
    totalFrameCount_ = inputVideo_.get(CV_CAP_PROP_FRAME_COUNT);
    //totalFrameCount_ = 100;
    
    frameWidth_ = inputVideo_.get(CV_CAP_PROP_FRAME_WIDTH);
    frameHeight_ = inputVideo_.get(CV_CAP_PROP_FRAME_HEIGHT);
    diagonal_ = (int)sqrt(frameHeight_*frameHeight_ + frameWidth_*frameWidth_);
    tauC_ = 0.1 * diagonal_;
    gamma_ = 0.5 * diagonal_;
}


void Hyperlapse::generateHypelapse(){
    calcFrameMatchingCost();
    
    pathSelection();
    
    stabilizeFrames();
    
    writeHyperlapse();
}


void Hyperlapse::calcFrameMatchingCost(){
    std::vector<std::vector<cv::KeyPoint>> keyPoints;
    Cm_ = cv::Mat::zeros(totalFrameCount_+1, totalFrameCount_+1, CV_32F);
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    std::vector<cv::Mat> descriptors;
    
    for (int fi=0; fi<totalFrameCount_; fi++){
        cv::Mat frame, frameTransposed;
        inputVideo_.read(frame);
        
        cv::transpose(frame, frameTransposed);
        inputFrames_.push_back(frameTransposed);
        
        cv::Mat descriptor;
        std::vector<cv::KeyPoint> keyPoint;
        detector->detectAndCompute(frame, cv::noArray(), keyPoint, descriptor);
        keyPoints.push_back(keyPoint);
        descriptors.push_back(descriptor);
        
        int ti = fi+1-window_;
        if (ti>=1){
            cv::BFMatcher matcher(cv::NORM_HAMMING);
            for (int tj = ti+1; tj <= std::min(ti+window_, totalFrameCount_); tj++){
                std::vector<cv::DMatch> matches;
                matcher.match(descriptors[0], descriptors[tj-ti-1], matches);
                Cm_.at<float>(ti, tj) = costFunction(matches,
                                                     keyPoints[ti-1],
                                                     keyPoints[tj-1]);
            }
            descriptors[0].release();
            descriptors.erase(descriptors.begin());
        }
    }
}


float Hyperlapse::costFunction(std::vector<cv::DMatch>& matches,
                               std::vector<cv::KeyPoint>& keyPointsI,
                               std::vector<cv::KeyPoint>& keyPointsJ){
    
    double minDist = DBL_MAX;
    for (int i=0; i<(int)matches.size(); i++){
        double dist = matches[i].distance;
        if(dist < minDist){
            minDist = dist;
        }
    }
    
    const double threshold = 3.0 * minDist;
    std::vector<cv::Point2f> matchesGoodI, matchesGoodJ;
    for (int i=0; i<(int)matches.size(); i++){
        if (matches[i].distance < threshold){
            matchesGoodI.push_back(keyPointsI[matches[i].queryIdx].pt);
            matchesGoodJ.push_back(keyPointsJ[matches[i].trainIdx].pt);
        }
    }
    
    if (matchesGoodI.size() < 4){
        return gamma_;
    }
    
    cv::Mat masks;
    cv::Mat homography = cv::findHomography(matchesGoodI, matchesGoodJ, CV_RANSAC, 3, masks);
    if (homography.rows == 0){
        return gamma_;
    }
    
    float Cr = reprojectionError(matchesGoodI, matchesGoodJ, homography);
    
    cv::Point2f x(frameWidth_/2), y(frameHeight_/2);
    std::vector<cv::Point2f> center = {x, y};
    float Co = reprojectionError(center, center, homography);
    
    if (Cr < tauC_){
        return Co;
    }else{
        return gamma_;
    }
}


float Hyperlapse::reprojectionError(std::vector<cv::Point2f> &keyPointsI,
                                    std::vector<cv::Point2f> &keyPointsJ,
                                    cv::Mat &homography){
    std::vector<cv::Point2f> keyPointsITransformed;
    cv::perspectiveTransform(keyPointsI, keyPointsITransformed, homography);
    
    int keiPointsCount = (int)keyPointsI.size();
    cv::Mat keyPointsITransformedMat = cv::Mat::zeros(keiPointsCount, 2, CV_32F);
    cv::Mat keyPointsJMat = cv::Mat::zeros(keiPointsCount, 2, CV_32F);
    
    for (int i=0; i<keiPointsCount; i++){
        keyPointsITransformedMat.at<float>(i, 0) = keyPointsITransformed[i].x;
        keyPointsITransformedMat.at<float>(i, 1) = keyPointsITransformed[i].y;
        keyPointsJMat.at<float>(i, 0) = keyPointsJ[i].x;
        keyPointsJMat.at<float>(i, 1) = keyPointsJ[i].y;
    }
    
    cv::Mat diff = keyPointsJMat - keyPointsITransformedMat;
    cv::Mat RSS;
    cv::reduce(diff.mul(diff), RSS, 1, CV_REDUCE_SUM);
    
    cv::Mat norm;
    cv::sqrt(RSS, norm);
    
    return cv::mean(norm)[0];
}


void Hyperlapse::pathSelection(){
    cv::Mat Dv = initialization(Cm_);
    
    cv::Mat Tv = cv::Mat::zeros(totalFrameCount_+1, totalFrameCount_+1, CV_32S);
    populateDv(Cm_, Dv, Tv);
    
    traceBackMinCostPath(Dv, Tv, inputFrames_, optimalFrames_);
}


cv::Mat Hyperlapse::initialization(cv::Mat Cm){
    cv::Mat Dv = cv::Mat::zeros(totalFrameCount_+1, totalFrameCount_+1, CV_32F);
    
    for (int fi=1; fi<gap_; fi++){
        for (int fj=fi+1; fj<fi+window_; fj++){
            float Cs = std::min(powf((fj-fi)-speedUpRate_, 2), (float) tauS_);
            Dv.at<float>(fi, fj) = Cm.at<float>(fi, fj) + lambdaS_* Cs;
        }
    }
    return Dv;
}


void Hyperlapse::populateDv(cv::Mat Cm, cv::Mat& Dv, cv::Mat& Tv){
    for (int fi=gap_; fi<totalFrameCount_; fi++){
        for (int fj=fi+1; fj<=std::min(fi+window_, totalFrameCount_); fj++){
            float Cs = std::min(powf((fj-fi)-speedUpRate_, 2), (float) tauS_);
            float c = Cm.at<float>(fi, fj) + lambdaS_ * Cs;
            
            double minK = FLT_MAX;
            int argminK = 0;
            for (int k=1; k<=std::min(fi-1, window_); k++){
                float Ca = std::min(powf((fj-fi)-speedUpRate_, 2), (float) tauA_);
                float value = Dv.at<float>(fi-k, fj) + lambdaA_ * Ca;
                if (value<minK){
                    minK = value;
                    argminK = k;
                }
            }
            Dv.at<float>(fi, fj) = c + minK;
            Tv.at<int>(fi, fj) = fi - argminK;
        }
    }
}


void Hyperlapse::traceBackMinCostPath(cv::Mat& Dv, cv::Mat& Tv,
                                      std::vector<cv::Mat> inputFrames,
                                      std::vector<cv::Mat>& optimalFrames){
    int s = 0;
    int d = 0;
    float minD = FLT_MAX;
    std::vector<int> optimalPath;
    for (int fi=totalFrameCount_-gap_; fi<=totalFrameCount_; fi++){
        for (int fj=fi+1; fj <= std::min(fi+window_, totalFrameCount_); fj++){
            float value = Dv.at<float>(fi, fj);
            if (value < minD){
                minD = value;
                s = fi;
                d = fj;
            }
        }
    }
    optimalPath.push_back(d);
    while(s>gap_){
        optimalPath.insert(optimalPath.begin(), s);
        int b = Tv.at<int>(s, d);
        d = s;
        s = b;
    }

    for(int index:optimalPath){
        optimalFrames.push_back(inputFrames[index]);
    }
}


void Hyperlapse::stabilizeFrames(){
    cv::videostab::StabilizerBase* stabilizer;
    cv::videostab::TwoPassStabilizer* twoPass = new cv::videostab::TwoPassStabilizer();
    twoPass->setMotionStabilizer(cv::makePtr<cv::videostab::GaussianMotionFilter>(15));
    stabilizer = twoPass;
    
    cv::Ptr<VecSource> source = cv::makePtr<VecSource>(& optimalFrames_);
    stabilizer->setFrameSource(source);
    
    cv::Ptr<cv::videostab::MotionEstimatorRansacL2> est = cv::makePtr<cv::videostab::MotionEstimatorRansacL2>(cv::videostab::MM_HOMOGRAPHY);
    cv::Ptr<cv::videostab::IOutlierRejector> outlierRejector = cv::makePtr<cv::videostab::NullOutlierRejector>();
    
    cv::Ptr<cv::videostab::KeypointBasedMotionEstimator> kbest = cv::makePtr<cv::videostab::KeypointBasedMotionEstimator>(est);
    kbest->setDetector(cv::GFTTDetector::create(1000));
    
    kbest->setOutlierRejector(outlierRejector);
    stabilizer->setMotionEstimator(kbest);
    
    stabilizer->setRadius(15);
    stabilizer->setTrimRatio(0.1);
    stabilizer->setBorderMode(cv::BORDER_REPLICATE);
    
    cv::Ptr<cv::videostab::WeightingDeblurer> deblurer = cv::makePtr<cv::videostab::WeightingDeblurer>();
    deblurer->setRadius(15);
    deblurer->setSensitivity(0.1);
    stabilizer->setDeblurer(deblurer);
    
    stabilizedFrames_.reset(dynamic_cast<cv::videostab::IFrameSource*>(stabilizer));
}


void Hyperlapse::writeHyperlapse(){
    cv::VideoWriter output;
    cv::Mat frame;
    int count = 0;
    while (!(frame = stabilizedFrames_->nextFrame()).empty()) {
        if (!output.isOpened()) {
            output.open(
                        outputVideoPath_,
                        CV_FOURCC('a','v','c','1'),
                        inputVideo_.get(CV_CAP_PROP_FPS),
                        frame.size());
        }
        output << frame;
        count++;
    }
}
