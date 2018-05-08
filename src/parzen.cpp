#include <opencv2/opencv.hpp>
#include "DataLoader.h"

using namespace std;
using namespace cv;

const bool IS_LOAD_ONCE = true;
const int N_KEY_POINTS = 300;
const double TRAIN_PERCENT = 0.8;
const int N_TOTAL_SET = 1000;
const Size SIZE_IMAGE = Size(6, 6);
const int WINDOW_SIZE = 2;

namespace test
{
    Mat preProcessMat(Mat in, Size size)
    {
        // Step 1: find the ROI of the image
        int l = in.cols, r = 0, t = in.rows, b = 0;
        for (int i = 0; i < in.cols; i++)
            for(int j = 0; j < in.rows; j++)
            {
                if(in.at<uchar>(i, j) > 0 )
                {
                    if(l > j) l = j;
                    if(r < j) r = j;
                    if(t > i) t = i;
                    if(b < i) b = i;
                }
            }
        Mat roi = in.adjustROI(t, b, l, r);

        // Step 2: resize the ROI into out size
        Mat out;
        resize(roi, out, size);
        return out;
    }


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
        for (int i = 0; i < 10; i ++)
        {
            cout << dataset[i].data_list.size() << endl;
        }
        for (int j = 0; j < 200; j++)
        {
            for (int i = 0; i < 10; i++)
            {
                Mat img = dataset[i].data_list[j+1].img;
                cout << img.cols << " | " << img.rows << endl;
                cout << img << endl;
                Mat img_test = preProcessMat(img, Size(5, 5));
                print(img_test);
                imshow("test", img);
                waitKey();
            }
        }
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

namespace ws
{
    Mat preProcessMat(Mat in, Size size = SIZE_IMAGE)
    {
        // Step 1: find the ROI of the image
        int l = in.cols, r = 0, t = in.rows, b = 0;
        for (int i = 0; i < in.cols; i++)
            for(int j = 0; j < in.rows; j++)
            {
                if(in.at<uchar>(i, j) > 0 )
                {
                    if(l > j) l = j;
                    if(r < j) r = j;
                    if(t > i) t = i;
                    if(b < i) b = i;
                }
            }
        Mat roi = in.adjustROI(t, b, l, r);

        // Step 2: resize the ROI into out size and make it to be a binary map
        Mat out, out_tmp;
        resize(roi, out_tmp, size);
        threshold(out_tmp, out, 2, 1, THRESH_BINARY);
        return out;
    }

    /**
     * @brief Advanced Parzen Window, determine whether a train point fell into the Window of Parzen made [beta]
     * @param sample    The sample point storage in a Mat
     * @param train     The train point, also storage in a Mat
     * @param width     The window width(size)
     * @param threshold Control the threshold of an image match another one
     * @return  0|1     Fell into the window, return 1, vise versa
     */
    int parzenWindowAdv(Mat sample, Mat train, int width, double threshold = 0.7)
    {
//        cout << "sample: " << endl;
//        cout << sample << endl;
//        cout << "train: " << endl;
//        cout << train << endl;
        int n_matched = 0;
        // To every pixel in train, search them in the sample with the window
        // if in the window there has a pixel has the same value with train, n_matched++
        for (int r = 0; r < train.rows; r++)
        {
            for (int c = 0; c < train.cols; c++)
            {
                bool b_matched = false;
                // Here, compare this pixel in the window
                for (int i = r - width; i < r + width; i++)
                {
                    for (int j = c - width; j < c + width; j++)
                    {
                        if(i >= 0 && i < sample.rows && j >= 0 && j < sample.cols)
                            if (sample.at<uchar>(i, j) == train.at<uchar>(r, c) && train.at<uchar>(r, c) == 1)
                            {
                                b_matched = true;
                                break;
                            }
                    }
                    if (b_matched) break;
                }
                // Then update the n_matched number
                if (b_matched) n_matched ++;
            }
        }
        // If pixel matched number
        if (n_matched > sample.cols * sample.rows * threshold * 2/5)
            return 1;
        else
            return 0;

    }

    /**
     * @brief Parzen Window, determine whether a train point fell into the Window of Parzen made
     * @param sample    The sample point storage in a Mat
     * @param train     The train point, also storage in a Mat
     * @param width     The window width(size)
     * @return  0|1     Fell into the window, return 1, vise versa
     */
    int parzenWindow(const Mat &sample, const Mat &train, int width)
    {
        Mat t_distance = sample - train;
        double distance = norm(t_distance, NORM_L2);

        if (distance < width)
            return 1;
        else
            return 0;
    }

    void parzenClassify()
    {
        // Step 1: Load the dataset from disk, here MNIST dataset is used
        cout << "#####################################" << endl;
        cout << "######### Loading dataset ..." << endl;
        cout << "#####################################" << endl;
        Mnist mnist;
        Dataset dataset = mnist.readDataset("../data/mnist");

        // Step 2: Pre-process dataset into train-set and test-set
        cout << "#####################################" << endl;
        cout << "######### Pre-processing the images..." << endl;
        cout << "#####################################" << endl;
        Dataset train_set, test_set;
        for (int i = 0; i < dataset.size(); i++)
        {
            Class train, test;
            train.class_id = dataset[i].class_id;
            test.class_id = dataset[i].class_id;

            for(int j = 0; j < N_TOTAL_SET * TRAIN_PERCENT && j < dataset[i].data_list.size(); j++)
            {
                Data data = dataset[i].data_list[j];
                data.img = preProcessMat(data.img);
                train.data_list.push_back(data);
            }
            train_set.push_back(train);

            for(int j = N_TOTAL_SET * TRAIN_PERCENT; j < N_TOTAL_SET && j < dataset[i].data_list.size(); j++)
            {
                Data data = dataset[i].data_list[j];
                data.img = preProcessMat(data.img);
                test.data_list.push_back(data);
            }
            test_set.push_back(test);
        }

        // Step 3: verify the test_set and print the error rate
        cout << "#####################################" << endl;
        cout << "######### Predict the test dataset..." << endl;
        cout << "#####################################" << endl;
        cout << "Using green(red) bounder to show this prediction is right(wrong)." << endl;
        float total_accuracy_rate = 0;
        for (int i = 0; i < test_set.size(); i++)
        {
            // Test each number and count their correct rate
            cout << "Testing the class: " << test_set[i].class_id << "  ===>";
            // When predict a number get right, this counter ++
            int n_accurate = 0;
            for (int j = 0; j < test_set[i].data_list.size(); ++j)
            {
                Mat sample = test_set[i].data_list[j].img;
                Mat show_img = dataset[i].data_list[j].img;
                cvtColor(show_img, show_img, CV_GRAY2BGR);

                // Compare the data with each class's data in train
                int post_class_id = -1, n_max_count = 0;
                for (int p = 0; p < train_set.size(); ++p)
                {
                    int n_count = 0;
                    for (int q = 0; q < train_set[p].data_list.size(); ++q)
                    {
                        Mat train = train_set[p].data_list[q].img;
                        // TODO: this number 10 need to be tuned
                        // Here resize as 6x6 and window width is 2 for parzenWindow is fine
                        n_count += parzenWindow(sample, train, 2);
//                        n_count += parzenWindowAdv(sample, train, 1);
                    }
                    if(n_count > n_max_count)
                    {
                        n_max_count = n_count;
                        post_class_id = train_set[p].class_id;
                    }
                }
                if (post_class_id == test_set[i].class_id)
                {
                    n_accurate++;
                    // Use green rectangle to show this prediction is right
                    rectangle(show_img, Point(0, 0), Point(28, 28), Scalar(0, 255, 0), 2);
                }
                else// Use red rectangle to show this prediction is wrong
                    rectangle(show_img, Point(0, 0), Point(28, 28), Scalar(0, 0, 255), 2);

                imshow("Prediction", show_img);
                waitKey(2);
            }
            float t_accuracy_rate = (float)n_accurate / (float)test_set[i].data_list.size();
            cout << " accuracy rate is: " << t_accuracy_rate * 100.0 << "%. " << endl;
            total_accuracy_rate = total_accuracy_rate + t_accuracy_rate;
        }
        cout << endl << "===================================================================" << endl;
        cout << "The test dataset's accuracy rate predicted by Parzen is: "
             << total_accuracy_rate /(float)test_set.size() * 100.0 << "%." << endl;

        destroyAllWindows();
    }
}

int main()
{

//    test::flower_load();
//    test::mnist_load();
//    test::orb_extract();
    ws::parzenClassify();

	return 0;
}