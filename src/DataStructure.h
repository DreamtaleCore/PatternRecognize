//
// Created by ros on 18-3-25.
//

#ifndef PR_DATASTRUCTURE_H
#define PR_DATASTRUCTURE_H
#include <vector>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
using namespace std;

enum DataSetType
{
    DatasetTrain = 0,
    DatasetValid = 1,
};

struct Data
{
    cv::Mat img;
    int label;
    string filename;
    vector<cv::KeyPoint> keypoints;
    DataSetType data_type;

    Data()  {img = cv::Mat(); label = -1; filename= ""; keypoints = vector<cv::KeyPoint>(0);
                data_type=DatasetTrain;}
    ~Data() {img.release(); filename = "";}
};

struct Class
{
    int class_id;
    vector<Data> data_list;

    Class() {data_list = vector<Data>(0); class_id = -1; }
};

typedef vector<Class> Dataset;
#endif //PR_DATASTRUCTURE_H
