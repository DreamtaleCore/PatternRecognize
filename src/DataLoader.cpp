//
// Created by ros on 18-3-24.
//

#include "DataLoader.h"

#include <dirent.h>
#include <sys/stat.h>
#include <fstream>
#include <sstream>

// region DataLoader functions

vector<string> DataLoader::get_all_files(string dir)
{
    vector<string> ret;
    struct stat s;
    lstat(dir.c_str(), &s);
    if (!S_ISDIR(s.st_mode))
    {
        cout << dir << " is not a valid directory !" << endl;
        return ret;
    }
    struct dirent* pfilename;
    DIR* pdir;
    pdir = opendir(dir.c_str());
    if (NULL == pdir)
    {
        cout << "Cannot open dir: " << dir << endl;
        return ret;
    }

    while ((pfilename = readdir(pdir)) != NULL)
    {
        // get rif of "." and ".."
        if (strcmp(pfilename->d_name, ".") == 0 ||
                strcmp(pfilename->d_name, "..") == 0)
            continue;
        ret.push_back(pfilename->d_name);
    }

    return ret;
}

/**
 * Load data from storage
 * @param dir
 * @param is_load   If is true, load image to memory, may be memory killer
 * @return          The structure of data
 */
Dataset DataLoader::readDataset(string dir, bool is_load)
{
    return Dataset();
}

void DataLoader::setTrainPercent(double percent)
{
    m_train_percent = percent;
}
// endregion

// region Flower17 functions

/**
 * Load data from storage for Flower17
 * @overload        DataLoader::readDataset
 * @param dir       Like mine: "/home/ros/ws/data_sets/17flowers"
 * @param is_load
 * @return
 */
Dataset Flower17::readDataset(string dir, bool is_load)
{
    Dataset ret;

    // Step 1: check the dir
    vector<string> dir_files = get_all_files(dir);
    bool is_right_dir = false;
    for(int i = 0; i < dir_files.size(); i++)
    {
        if (dir_files[i] == "jpg")
        {
            is_right_dir = true;
            break;
        }
    }
    if(!is_right_dir)
    {
        cout << dir << " is not the flower17's directory!" << endl;
        return ret;
    }
    cout << "The dir is ok, Loading..." << endl;

    // Step2: Load it!
    string jpg_dir = dir;
    if (*(jpg_dir.end() - 1) != '/')
        jpg_dir += '/';
    jpg_dir += "jpg";

    vector<string> jpg_files = get_all_files(jpg_dir);
    for (int i = 0; i < jpg_files.size(); i++)
    {
        string class_dir = jpg_dir + "/" + jpg_files[i];
        struct stat s;
        lstat(class_dir.c_str(), &s);
        // Ensure every class is a directory
        if(!S_ISDIR(s.st_mode))
            continue;
        // The directory name is the class id
        stringstream ss;
        ss << jpg_files[i];
        int class_id = -1;
        ss >> class_id;
        vector<string> img_files = get_all_files(class_dir);
        Class a_class;
        a_class.class_id = class_id;
        for (int j = 0; j <img_files.size() ; j++)
        {
            string img_path = class_dir + "/" + img_files[j];
            Data data;
            if(is_load)
            {
                data.img = cv::imread(img_path);
            }
            data.filename = img_path;
            data.label = class_id;
            if (j < img_files.size() * m_train_percent)
                data.data_type = DatasetTrain;
            else
                data.data_type = DatasetValid;

            a_class.data_list.push_back(data);
        }
        ret.push_back(a_class);
    }
    cout << "Load dataset successfully. " << endl;
    return ret;
}

// endregion

// region Mnist functions

/**
 * Load data from storage for MNIST
 * @overload        DataLoader::readDataset
 * @param dir       Like mine: "/home/ros/ws/data_sets/mnist"
 * @param is_load
 * @return
 */
Dataset Mnist::readDataset(string dir, bool is_load)
{

    Dataset ret;

    // Step 1: check the dir
    vector<string> dir_files = get_all_files(dir);
    int is_right_dir = 0;
    for(int i = 0; i < dir_files.size(); i++)
    {
        if (dir_files[i] == "t10k-images.idx3-ubyte")
            is_right_dir++;
        if (dir_files[i] == "t10k-labels.idx1-ubyte")
            is_right_dir++;
        if(dir_files[i] == "train-images.idx3-ubyte")
            is_right_dir++;
        if(dir_files[i] == "train-labels.idx1-ubyte")
            is_right_dir++;
    }
    if(is_right_dir != 4)
    {
        cout << dir << " is not the MNIST's directory!" << endl;
        return ret;
    }

    // Step2: Load it!
    vector<int> labels_t, labels_v;
    vector<vector<unsigned char> > images_t, images_v;
    read_labels(dir + "/" + "train-labels.idx1-ubyte", labels_t);
    read_labels(dir + "/" + "t10k-labels.idx1-ubyte",  labels_v);
    read_images(dir + "/" + "train-images.idx3-ubyte", images_t);
    read_images(dir + "/" + "t10k-images.idx3-ubyte",  images_v);

    // Step3: Format the images & labels into Dataset
    //        Double check the data set is whether valid or not
    cout << labels_t.size() << endl;
    cout << labels_v.size() << endl;
    if (labels_t.size() != images_t.size() ||
            labels_v.size() != images_v.size())
    {
        cout << "Dataset dir: " << dir << "'s data is not align" << endl;
        return ret;
    }
    cout << "The dir is ok, Loading..." << endl;

    int max_label = -1;
    for (int i = 0; i < labels_v.size(); i++)
    {
        if (max_label < labels_v[i])
            max_label = labels_v[i];
    }
    ret.clear();
    ret = vector<Class>(static_cast<unsigned long>(max_label + 1));
    for (int i = 0; i < images_t.size(); i++)
    {
        Data data;
        data.label = labels_t[i];
        data.img = cv::Mat(m_n_rows, m_n_cols, CV_8UC1);
        memcpy(data.img.data, images_t[i].data(), images_t[i].size() * sizeof(unsigned char));
        data.data_type = DatasetTrain;

        ret[labels_t[i]].class_id = labels_t[i];
        ret[labels_t[i]].data_list.push_back(data);
    }
    for (int i = 0; i < images_v.size(); i++)
    {
        Data data;
        data.label = labels_v[i];
        data.img = cv::Mat(m_n_rows, m_n_cols, CV_8UC1);
        memcpy(data.img.data, images_v[i].data(), images_v[i].size() * sizeof(unsigned char));
        data.data_type = DatasetValid;

        ret[labels_v[i]].class_id = labels_v[i];
        ret[labels_v[i]].data_list.push_back(data);
    }

    return ret;
}

int Mnist::reverse_int(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = static_cast<unsigned char>(i & 255);
    ch2 = static_cast<unsigned char>((i >> 8) & 255);
    ch3 = static_cast<unsigned char>((i >> 16) & 255);
    ch4 = static_cast<unsigned char>((i >> 24) & 255);
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void Mnist::read_labels(string dir, vector<int> &labels)
{
    ifstream in(dir.c_str(), ios::binary);
    if (in.is_open())
    {
        int magic_number = 0;
        int n_images = 0;
        in.read((char*)&magic_number, sizeof(magic_number));
        in.read((char*)&n_images, sizeof(magic_number));
        // resume int from binaries
        magic_number = reverse_int(magic_number);
        n_images     = reverse_int(n_images);

        cout << "Magic number: " << magic_number << endl;
        cout << "Number of images: " << n_images << endl;

        for (int i = 0; i < n_images; i++)
        {
            unsigned char label = 0;
            in.read((char*)& label, sizeof(label));
            labels.push_back(label);
        }
    }
}

void Mnist::read_images(string dir, vector<vector<unsigned char> > &images)
{
    ifstream in(dir.c_str(), ios::binary);
    if (in.is_open())
    {
        int magic_number = 0;
        int n_images = 0;
        int n_cols = 0;
        int n_rows = 0;
        in.read((char*)&magic_number, sizeof(magic_number));
        in.read((char*)&n_images, sizeof(n_images));
        in.read((char*)&n_rows, sizeof(n_rows));
        in.read((char*)&n_cols, sizeof(n_cols));
        // resume int from binaries
        magic_number = reverse_int(magic_number);
        n_images     = reverse_int(n_images);
        n_rows       = reverse_int(n_rows);
        n_cols       = reverse_int(n_cols);
        m_n_cols     = n_cols;
        m_n_rows     = n_rows;

        cout << "Magic number: " << magic_number << endl;
        cout << "Number of images: " << n_images << endl;
        cout << "n_cols: " << n_cols << endl;
        cout << "n_rows: " << n_rows << endl;

        for (int i = 0; i < n_images; i++)
        {
            vector<unsigned char> image;
            for (int r = 0; r < n_rows; r++)
            {
                for (int c = 0; c < n_cols; c++)
                {
                    unsigned char pixel;
                    in.read((char*)&pixel, sizeof(pixel));
                    image.push_back(pixel);
                }
            }
            images.push_back(image);

        }
    }
}

// endregion