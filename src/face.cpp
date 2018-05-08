#include <utility>

#include "DataLoader.h"
#include <ctime>

const int N_REDUCE_DIMENTATION = 800;
const string DISTANCE_TYPE = "Mahalanobis";
const bool IS_SHOW_FACE = false;
const string IMAGE_SAVE_DIR = "/home/ros/ws/algorithm/PatternRecognize/data/ORL_data_out/eigen_face_100";

void showImage(string title, cv::Mat image, bool is_save = false)
{
    cv::namedWindow(title);
    cv::moveWindow(title, 500, 500);

    cv::Mat image_show;
    if(image.type() != CV_8U)
        image.convertTo(image_show, CV_8U);
    else
        image_show = image;

    cv::imshow(title, image_show);
    if (is_save)
        cv::imwrite(IMAGE_SAVE_DIR + "/" + title + ".bmp", image_show);
}

void createDataMatrix(Dataset dataset, cv::Mat& total_mat, vector<cv::Mat>& class_mat)
{
    cout << "Creating data matrix from images ...";
    class_mat.clear();

    // Get the nou of all images
    int n_images = 0;
    for(int i = 0; i < dataset.size(); i++)
        n_images += dataset[i].data_list.size();

    cv::Mat total_img_mat(
            n_images,
            dataset[0].data_list[0].img.cols * dataset[0].data_list[0].img.rows,
            CV_32F
    );
    int image_idx = 0;

    for(int i = 0; i < dataset.size(); i++)
    {
        cv::Mat class_mat_i(
                static_cast<int>(dataset[i].data_list.size()),
                dataset[i].data_list[0].img.cols * dataset[i].data_list[0].img.rows,
                CV_32F
        );
        for (int j = 0; j < dataset[i].data_list.size(); j++)
        {
            cv::Mat image_gray;
            if(dataset[i].data_list[j].img.channels() != 1)
                cv::cvtColor(dataset[i].data_list[j].img, image_gray, CV_BGR2GRAY);
            else
                image_gray = dataset[i].data_list[j].img;

            cv::Mat image = image_gray.reshape(0, 1);
            image.copyTo(class_mat_i.row(j));
            image.copyTo(total_img_mat.row(image_idx++));
        }

        class_mat.push_back(class_mat_i);
    }

    total_mat = total_img_mat.clone();
    cout << "done." << endl;
}

cv::Mat computeWeightVector(cv::Mat img, const cv::Mat &mean, cv::Mat eigen_vectors)
{

    cv::Mat img_ori = std::move(img);
    if (img_ori.channels() != 1)
        cv::cvtColor(img_ori, img_ori, CV_BGR2GRAY);
    cv::Mat img_ori_vec_raw = img_ori.reshape(0, 1), img_ori_vec;
    img_ori_vec_raw.convertTo(img_ori_vec, CV_32F);

    cv::Mat weight_vec(eigen_vectors.rows, 1, CV_32F);
    for (int i = 0; i < weight_vec.rows; i++)
    {
        cv::Mat tmp_1 = img_ori_vec - mean;
        weight_vec.at<float>(i, 0) = static_cast<float>(eigen_vectors.row(i).dot(tmp_1));
    }

    return weight_vec;
}

double computeDistance(const cv::Mat &a, const cv::Mat &b, const string &method="Euclidean")
{
    double ret;
    if (method == "Manhattan")
    {
        ret = cv::norm(a-b, cv::NORM_L1);
    }
    else if (method == "Mahalanobis")
    {
        cv::Mat a_m, a_sd, b_m, b_sd;
        cv::meanStdDev(a, a_m, a_sd);
        cv::meanStdDev(b, b_m, b_sd);

        cv::Mat a_tmp = a / a_sd.at<double>(0, 0);
        cv::Mat b_tmp = b / b_sd.at<double>(0, 0);

        ret = cv::norm(a_tmp - b_tmp, cv::NORM_L2);
    }
    else
    {
        ret = cv::norm(a-b, cv::NORM_L2);
    }
    return ret;
}

namespace test
{


    void load_data()
    {
        Face face;
        Dataset dataset = face.readDataset(
                "/home/ros/ws/algorithm/PatternRecognize/data/ORL_face_dataset/ORL92112/bmp",
                "bmp"
        );

//        cv::namedWindow("test");
//        cv::moveWindow("test", 500, 500);
//        cv::imshow("test", dataset[0].data_list[0].img);
//        cv::waitKey();

        cv::Mat total_mat;
        vector<cv::Mat> class_mat;
        createDataMatrix(dataset, total_mat, class_mat);

        cout << "calc PCA ..." << endl;
        cv::Size sz = dataset[0].data_list[0].img.size();

        cv::PCA pca(total_mat, cv::Mat(), cv::PCA::DATA_AS_ROW, 500);
        cout << "done." << endl;

        cv::Mat average_face = pca.mean.reshape(0, sz.height);

        cv::Mat eigen_vectors = pca.eigenvectors;

        // compute the weight vector
        cv::Mat img_ori = dataset[0].data_list[0].img;
        if (img_ori.channels() != 1)
            cv::cvtColor(img_ori, img_ori, CV_BGR2GRAY);
        cv::Mat img_ori_vec_raw = img_ori.reshape(0, 1), img_ori_vec;
        img_ori_vec_raw.convertTo(img_ori_vec, 21);

        cv::Mat weight_vec(eigen_vectors.rows, 1, 21);
        for (int i = 0; i < weight_vec.rows; i++)
        {
            cout << img_ori_vec.size() << endl;
            cout << pca.mean.size() << endl;
            cout << img_ori_vec.channels() << endl;
            cout << img_ori_vec.type() << endl;
            cout << pca.mean.type() << endl;
            cv::Mat tmp_1 = img_ori_vec - pca.mean;
            weight_vec.at<float>(i, 0) = static_cast<float>(eigen_vectors.row(i).dot(tmp_1));
        }

        // refact the image through weight_vec
        cv::Mat img_ref = cv::Mat::zeros(img_ori_vec.size(), CV_32F);
        for(int i = 0; i < weight_vec.rows; i++)
        {
            img_ref = img_ref + weight_vec.at<float>(i, 0) * eigen_vectors.row(i);
        }
        img_ref = img_ref + pca.mean;

        showImage("original", img_ori);
        showImage("refactor", img_ref.reshape(0, sz.height));


        cv::waitKey();

        cout << "ok"<< endl;
    }
}

int main()
{
//    test::load_data();
    // Step 1: load train dataset
    Face face;
    Dataset dataset_train = face.readDataset(
            "/home/ros/ws/algorithm/PatternRecognize/data/ORL_face_dataset/ORL92112_modified/train",
            "bmp"
    );
    cv::Size img_sz = dataset_train[0].data_list[0].img.size();

    // Step 2: convert dataset into big matrix
    cv::Mat total_mat;
    vector<cv::Mat> class_mats;
    createDataMatrix(dataset_train, total_mat, class_mats);

    // Step 3: compute the PCA of total dataset's images & PCA of each class's images
    cout << "Calculate total train dataset's PCA ..." << endl;

    cv::PCA pca_total(total_mat, cv::Mat(), cv::PCA::DATA_AS_ROW, N_REDUCE_DIMENTATION);
    cout << "done." << endl;

    vector<cv::PCA> pca_classes;
    for (int i = 0; i < class_mats.size(); i++)
    {
        cout << "-- processing " << i << "/" << class_mats.size() << " class's PCA ..." << endl;
        cv::PCA pca_class(class_mats[i], cv::Mat(), cv::PCA::DATA_AS_ROW, N_REDUCE_DIMENTATION);
        pca_classes.push_back(pca_class);
    }
    cout << "done." << endl;

    // Step 4: compute the weight vector of each class's PCA.mean
    vector<cv::Mat> classes_weights;
    for (int i = 0; i < pca_classes.size(); i++)
    {
        cv::Mat weight = computeWeightVector(pca_classes[i].mean, pca_total.mean, pca_total.eigenvectors);

        classes_weights.push_back(weight);
    }

    // Step 4.5 visualize class eigen face for bettter debug
    for (int i = 0; i < pca_classes.size(); ++i)
    {
        stringstream ss;
        ss << "class " << dataset_train[i].class_id;
        if (IS_SHOW_FACE)
        {
            showImage(ss.str(), pca_classes[i].mean.reshape(0, img_sz.height), false);
            cv::waitKey(50);
        }
    }

    // Step 5: load test dataset
    cout << endl << "===== Begin to test =====" << endl;
    Dataset dataset_test = face.readDataset(
            "/home/ros/ws/algorithm/PatternRecognize/data/ORL_face_dataset/ORL92112_modified/test",
            "bmp"
    );

    // Step 6: compute each image's weight vector through pca_total
    double t_start = clock();
    int n_right = 0, n_total = 0;
    for (int i = 0; i < dataset_test.size(); i++)
    {
        int n_right_k = 0;
        int n_total_k = static_cast<int>(dataset_test[i].data_list.size());
        n_total += n_total_k;
        for (int j = 0; j < dataset_test[i].data_list.size(); j++)
        {
            cv::Mat weight_k = computeWeightVector(dataset_test[i].data_list[j].img,
                                                   pca_total.mean,
                                                   pca_total.eigenvectors);
            // Step 6.5 Compare the distance of train set's classes weight and find the nearest one as result
            double min_distance = 1e20;
            int class_id_out = -1;
            for (int k = 0; k < classes_weights.size(); k++)
            {
                double distance = computeDistance(weight_k, classes_weights[k], DISTANCE_TYPE);

                if(distance < min_distance)
                {
                    min_distance = distance;
                    class_id_out = dataset_test[k].class_id;
                }
            }
            if(class_id_out == dataset_test[i].data_list[j].label)
            {
                n_right++;
                n_right_k++;
            }
        }

        cout << "In test class \t" << dataset_test[i].class_id << ": "
             << n_right_k << "/" << n_total_k <<"(right/total) \t| right rate="
             << (double)n_right_k / (double)n_total_k << endl;
    }

    cout << "============================================" << endl;
    cout << "In test dataset: "
         << n_right << "/" << n_total <<"(right/total) | right rate="
         << (double)n_right / (double)n_total << endl;
    cout << "Elapsed time: " << ((clock() - t_start) / CLOCKS_PER_SEC) << " s." << endl;
    cout << "###############################################################" << endl;


    return 0;
}