//
// Created by ros on 18-3-24.
//

#ifndef PR_FLOWERS17_H
#define PR_FLOWERS17_H

#include "DataStructure.h"

class DataLoader
{
public:
    virtual Dataset readDataset(string dir, bool is_load = true);
    void setTrainPercent(double percent);
    DataLoader() {m_train_percent = 0.8;}

protected:
    vector<string> get_all_files(string dir);

protected:
    double m_train_percent;
};

class Flower17 : public DataLoader
{
public:
    Dataset readDataset(string dir, bool is_load = true);

};


class Mnist : public DataLoader
{
public:
    Dataset readDataset(string dir, bool is_load = true);

private:
    int reverse_int(int i);
    void read_labels(string dir, vector<int>& labels);
    void read_images(string dir, vector<vector<unsigned char> >& images);

private:
    int m_n_rows;
    int m_n_cols;
};

#endif //PR_FLOWERS17_H
