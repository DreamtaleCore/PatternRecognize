#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
	Mat img = imread("/home/ros/Pictures/girl1.jpg");

	imshow("test", img);
	waitKey();

	return 0;
}