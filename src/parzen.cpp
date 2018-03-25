#include <opencv2/opencv.hpp>
#include "DataLoader.h"

using namespace std;
using namespace cv;

const bool IS_LOAD_ONCE = true;
const int N_KEY_POINTS = 300;
const double TRAIN_PERCENT = 0.8;

namespace test
{
    void flower_load()
    {
        Flower17 flowers;
        Dataset dataset = flowers.readDataset("/home/ros/ws/data_sets/17flowers", IS_LOAD_ONCE);
        cout << dataset.size() << endl;
        cout << dataset[1].class_id << endl;
        imshow("test", dataset[1].data_list[1].img);
        waitKey();
    }

    void mnist_load()
    {
        Mnist mnist;
        Dataset dataset = mnist.readDataset("/home/ros/ws/data_sets/mnist");

        cout << dataset.size() << endl;
        Mat img = dataset[1].data_list[0].img;
        cout << img.cols << " | " << img.rows << endl;
        cout << img << endl;
        imshow("test", img);
        waitKey();
    }

    void feature_extract()
    {

    }

    void orb_extract()
    {
        Mat img1 = imread("/home/ros/ws/data_sets/17flowers/jpg/1/image_0089.jpg");
        Mat img2 = imread("/home/ros/ws/data_sets/17flowers/jpg/12/image_0974.jpg");
        Ptr<ORB> orb = ORB::create(50);
        vector<KeyPoint> kps1, kps2;
        Mat descriptors1, descriptors2;
        orb->detectAndCompute(img1, Mat(), kps1, descriptors1);
        orb->detectAndCompute(img2, Mat(), kps2, descriptors2);
        imshow("desp1", descriptors1);
        imshow("desp2", descriptors2);

        cout << kps1.size() << endl;
        Mat show1, show2;
        drawKeypoints(img1, kps1, show1);
        drawKeypoints(img2, kps2, show2);
        imshow("Key points 1", show1);
        imshow("Key points 2", show2);
        waitKey();

        vector<DMatch> matches;
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
        matcher->match(descriptors1, descriptors2, matches);
        cout << "Find total: " << matches.size() << " matches" << endl;

        Mat show;
        drawMatches(img1, kps1, img2, kps2, matches, show);
        imshow("matches", show);
        waitKey();
    }
}

int main()
{

//    test::flower_load();
    test::mnist_load();
//    test::orb_extract();

	return 0;
}