#ifndef MAT_HPP
#define MAT_HPP
#include <vector>
#include <functional>
#include <cmath>
#include <ctime>
#include <string>
#include <sstream>
#include <iostream>
#include "assert.h"
namespace RL {

class Mat
{
public:
    using ValueType = float;
    class Size
    {
    public:
        std::size_t rows;
        std::size_t cols;
    public:
        Size():rows(0),cols(0){}
        explicit Size(std::size_t r, std::size_t c)
            :rows(r),cols(c){}
    };

public:
    std::size_t rows;
    std::size_t cols;
    std::size_t totalSize;
    std::vector<float> val;
public:
    Mat(){}
    explicit Mat(std::size_t rows_, std::size_t cols_)
        :rows(rows_),cols(cols_),totalSize(rows_*cols_),val(rows_*cols_, 0){}
    explicit Mat(std::size_t rows_, std::size_t cols_, float value)
        :rows(rows_),cols(cols_),totalSize(rows_*cols_),val(rows_*cols_, value){}
    explicit Mat(const Size &s)
        :rows(s.rows),cols(s.cols),totalSize(s.rows*s.cols),val(s.rows*s.cols, 0){}
    explicit Mat(std::size_t rows_, std::size_t cols_, const std::vector<float> &data_)
        :rows(rows_),cols(cols_),totalSize(rows_*cols_),val(data_){}
    Mat(const Mat &r)
        :rows(r.rows),cols(r.cols),totalSize(r.totalSize),val(r.val){}
    Mat(Mat &&r)
        :rows(r.rows),cols(r.cols),totalSize(r.totalSize)
    {
        val.swap(r.val);
        r.rows = 0;
        r.cols = 0;
        r.totalSize = 0;
    }
    static void fromArray(const std::vector<Mat> &x, Mat &y)
    {
        /* x: (N, 1, featureDim) */
        y = Mat(x.size(), x[0].cols);
        for (std::size_t i = 0; i < y.rows; i++) {
            for (std::size_t j = 0; j < y.cols; j++) {
                y(i, j) = x[i](0, j);
            }
        }
        return;
    }

    static Mat zeros(std::size_t rows, std::size_t cols)
    {
        return Mat(rows, cols);
    }

    static Mat identity(std::size_t rows, std::size_t cols)
    {
        Mat y(rows, cols);
        std::size_t N = std::min(rows, cols);
        for (std::size_t i = 0; i < N; i++) {
            y(i, i) = 1;
        }
        return y;
    }
    /* size */
    inline std::size_t size() const {return totalSize;}
    /* visit */
    inline float &operator[](std::size_t i){return val[i];}
    inline float operator[](std::size_t i) const {return val[i];}
    inline float operator()(std::size_t i, std::size_t j) const {return val[i*cols + j];}
    inline float &operator()(std::size_t i, std::size_t j) {return val[i*cols + j];}
    /* iterator */
    inline std::vector<ValueType>::iterator begin() {return val.begin();}
    inline std::vector<ValueType>::iterator end() {return val.end();}
    Mat &operator=(const Mat &r)
    {
        if (this == &r) {
            return *this;
        }
        rows = r.rows;
        cols = r.cols;
        totalSize = r.totalSize;
        if (val.empty()) {
            val = std::vector<float>(r.val);
        } else {
            val.assign(r.val.begin(), r.val.end());
        }
        return *this;
    }

    Mat &operator=(float value)
    {
        for (std::size_t i = 0; i < val.size(); i++) {
            val[i] = value;
        }
        return *this;
    }

    Mat &operator=(Mat &&r)
    {
        if (this == &r) {
            return *this;
        }
        rows = r.rows;
        cols = r.cols;
        totalSize = r.totalSize;
        val.swap(r.val);
        r.rows = 0;
        r.cols = 0;
        r.totalSize = 0;
        return *this;
    }

    Mat operator *(const Mat &x) const
    {
        if (cols != x.rows) {
            return Mat();
        }
        Mat y(rows, x.cols);
        for (std::size_t i = 0; i < y.rows; i++) {
            for (std::size_t j = 0; j < y.cols; j++) {
                for (std::size_t k = 0; k < cols; k++) {
                    y.val[i*y.cols + j] += val[i*cols + k] * x.val[k*x.cols + j];
                }
            }
        }
        return y;
    }

    Mat operator /(const Mat &x) const
    {
        Mat y(rows, cols);
        for (std::size_t i = 0; i < y.val.size(); i++) {
            y.val[i] = val[i]/x.val[i];
        }
        return y;
    }

    Mat operator +(const Mat &x) const
    {
        Mat y(rows, cols);
        for (std::size_t i = 0; i < y.val.size(); i++) {
            y.val[i] = val[i] + x.val[i];
        }
        return y;
    }

    Mat operator -(const Mat &x) const
    {
        Mat y(rows, cols);
        for (std::size_t i = 0; i < y.val.size(); i++) {
            y.val[i] = val[i] - x.val[i];
        }
        return y;
    }

    Mat operator %(const Mat &x) const
    {
        Mat y(rows, cols);
        for (std::size_t i = 0; i < y.val.size(); i++) {
            y.val[i] = val[i] * x.val[i];
        }
        return y;
    }

    Mat &operator /=(const Mat &x)
    {
        for (std::size_t i = 0; i < val.size(); i++) {
            val[i] /= x.val[i];
        }
        return *this;
    }

    Mat &operator +=(const Mat &x)
    {
        for (std::size_t i = 0; i < val.size(); i++) {
            val[i] += x.val[i];
        }
        return *this;
    }

    Mat &operator -=(const Mat &x)
    {
        for (std::size_t i = 0; i < val.size(); i++) {
            val[i] -= x.val[i];
        }
        return *this;
    }

    Mat &operator %=(const Mat &x)
    {
        for (std::size_t i = 0; i < val.size(); i++) {
            val[i] *= x.val[i];
        }
        return *this;
    }

    Mat operator *(float x) const
    {
        Mat y(rows, cols);
        for (std::size_t i = 0; i < val.size(); i++) {
            y.val[i] = val[i] * x;
        }
        return y;
    }

    Mat operator /(float x) const
    {
        Mat y(rows, cols);
        for (std::size_t i = 0; i < val.size(); i++) {
            y.val[i] = val[i] / x;
        }
        return y;
    }

    Mat operator +(float x) const
    {
        Mat y(rows, cols);
        for (std::size_t i = 0; i < val.size(); i++) {
            y.val[i] = val[i] + x;
        }
        return y;
    }

    Mat operator -(float x) const
    {
        Mat y(rows, cols);
        for (std::size_t i = 0; i < val.size(); i++) {
            y.val[i] = val[i] - x;
        }
        return y;
    }

    Mat &operator *=(float x)
    {
        for (std::size_t i = 0; i < val.size(); i++) {
            val[i] *= x;
        }
        return *this;
    }

    Mat &operator /=(float x)
    {
        for (std::size_t i = 0; i < val.size(); i++) {
            val[i] /= x;
        }
        return *this;
    }

    Mat &operator +=(float x)
    {
        for (std::size_t i = 0; i < val.size(); i++) {
            val[i] += x;
        }
        return *this;
    }

    Mat &operator -=(float x)
    {
        for (std::size_t i = 0; i < val.size(); i++) {
            val[i] -= x;
        }
        return *this;
    }

    Mat tr() const
    {
        Mat y(cols, rows);
        for (std::size_t i = 0; i < y.rows; i++) {
            for (std::size_t j = 0; j < y.cols; j++) {
                y.val[i*y.cols + j] = val[j*cols + i];
            }
        }
        return y;
    }

    int reshape(size_t rows_, size_t cols_)
    {
        if (rows_*cols_ != totalSize) {
            return -1;
        }
        rows = rows_;
        cols = cols_;
        return 0;
    }

    Mat sub(std::size_t pr, std::size_t pc, std::size_t sr, std::size_t sc) const
    {
        Mat y(sr, sc);
        for (std::size_t i = 0; i < y.rows; i++) {
            for (std::size_t j = 0; j < y.cols; j++) {
                y.val[i*y.cols + j] = val[(i + pr)*cols + j + pc];
            }
        }
        return y;
    }

    Mat flatten() const
    {
        Mat y(rows*cols, 1, val);
        return y;
    }

    Mat f(std::function<float(float)> func) const
    {
        Mat y(rows, cols);
        for (std::size_t i = 0; i < val.size(); i++) {
            y.val[i] = func(val[i]);
        }
        return y;
    }

    void set(std::size_t pr, std::size_t pc, const Mat &x)
    {
        for (std::size_t i = pr; i < pr + x.rows; i++) {
            for (std::size_t j = pc; j < pc + x.cols; j++) {
                val[i*cols + j] = x.val[(i - pr)*cols + j - pc];
            }
        }
        return;
    }

    void setRow(size_t i, const std::vector<float> &row)
    {
        for (std::size_t j = 0; j < cols; j++) {
            val[i*cols + j] = row[j];
        }
        return;
    }

    void setColumn(size_t j, const std::vector<float> &col)
    {
        for (std::size_t i = 0; i < rows; i++) {
            val[i*cols + j] = col[i];
        }
        return;
    }

    void zero()
    {
        val.assign(totalSize, 0);
        return;
    }

    void fill(float x)
    {
        val.assign(totalSize, x);
        return;
    }

    void show() const
    {
        //std::cout<<"row:"<<rows<<", cols:"<<cols<<std::endl;
        for (std::size_t i = 0; i < rows; i++) {
            for (std::size_t j = 0; j < cols; j++) {
                std::size_t index = i*cols + j;
                std::cout<<val[index]<<" ";
            }
            std::cout<<std::endl;
        }
        return;
    }

    size_t argmax() const
    {
        float maxValue = val[0];
        std::size_t index = 0;
        for (std::size_t i = 1; i < totalSize; i++) {
            if (val[i] > maxValue) {
                maxValue = val[i];
                index = i;
            }
        }
        return index;
    }

    size_t argmin() const
    {
        float minValue = val[0];
        std::size_t index = 0;
        for (std::size_t i = 1; i < totalSize; i++) {
            if (val[i] < minValue) {
                minValue = val[i];
                index = i;
            }
        }
        return index;
    }

    float max() const
    {
        float maxValue = val[0];
        for (std::size_t i = 1; i < totalSize; i++) {
            if (val[i] > maxValue) {
                maxValue = val[i];
            }
        }
        return maxValue;
    }

    float min() const
    {
        float minValue = val[0];
        for (std::size_t i = 1; i < totalSize; i++) {
            if (val[i] < minValue) {
                minValue = val[i];
            }
        }
        return minValue;
    }

    float sum() const
    {
        float s = 0;
        for (std::size_t i = 0; i < totalSize; i++) {
            s += val[i];
        }
        return s;
    }

    float mean() const
    {
        return sum()/float(totalSize);
    }

    float variance() const
    {
        float u = mean();
        float s = 0;
        for (std::size_t i = 0; i < totalSize; i++) {
            s += (val[i] - u)*(val[i] - u);
        }
        return std::sqrt(s/float(totalSize));
    }

    static Mat kronecker(const Mat &x1, const Mat &x2)
    {
        Mat y(x1.rows * x2.rows, x1.cols * x2.cols);
        for (std::size_t i = 0; i < x1.rows; i++) {
            for (std::size_t j = 0; j < x1.cols; j++) {
                for (std::size_t k = 0; k < x2.rows; k++) {
                    for (std::size_t l = 0; l < x2.cols; l++) {
                        y(i*x2.cols + k, j*x2.cols + l) = x1.val[i*x1.cols + j] * x2.val[k*x2.cols + l];
                    }
                }
            }
        }
        return y;
    }
    struct Swap {
        static void row(Mat &x, size_t ri, size_t rj)
        {
            for (std::size_t h = 0; h < x.cols; h++) {
                float tmp = x(ri, h);
                x(ri, h) = x(rj, h);
                x(rj, h) = tmp;
            }
            return;
        }

        static void col(Mat &x, size_t ci, size_t cj)
        {
            for (std::size_t h = 0; h < x.rows; h++) {
                float tmp = x(h, ci);
                x(h, ci) = x(h, cj);
                x(h, cj) = tmp;
            }
            return;
        }
    };
    struct Multiply {
        static void ikkj(Mat &y, const Mat &x1, const Mat &x2)
        {
            /* no transpose */
            for (std::size_t i = 0; i < y.rows; i++) {
                for (std::size_t j = 0; j < y.cols; j++) {
                    for (std::size_t k = 0; k < x1.cols; k++) {
                        /* (i, j) = (i, k) * (k, j) */
                        y.val[i*y.cols + j] += x1.val[i*x1.cols + k]*x2.val[k*x2.cols + j];
                    }
                }
            }
            return;
        }

        static void ikjk(Mat &y, const Mat &x1, const Mat &x2)
        {
            /* transpose x2 */
            for (std::size_t i = 0; i < y.rows; i++) {
                for (std::size_t j = 0; j < y.cols; j++) {
                    for (std::size_t k = 0; k < x1.cols; k++) {
                        /* (i, j) = (i, k) * (j, k)^T */
                        y.val[i*y.cols + j] += x1.val[i*x1.cols + k]*x2.val[j*x2.cols + k];
                    }
                }
            }
            return;
        }

        static void kikj(Mat &y, const Mat &x1, const Mat &x2)
        {
            /* transpose x1 */
            for (std::size_t i = 0; i < y.rows; i++) {
                for (std::size_t j = 0; j < y.cols; j++) {
                    for (std::size_t k = 0; k < x1.rows; k++) {
                        /* (i, j) = (k, i)^T * (k, j)^T */
                        y.val[i*y.cols + j] += x1.val[k*x1.cols + i]*x2.val[k*x2.cols + j];
                    }
                }
            }
            return;
        }

        static void kijk(Mat &y, const Mat &x1, const Mat &x2)
        {
            /* transpose x1, x2 */
            for (std::size_t i = 0; i < y.rows; i++) {
                for (std::size_t j = 0; j < y.cols; j++) {
                    for (std::size_t k = 0; k < x1.rows; k++) {
                        /* (i, j) = (k, i)^T * (j, k)^T */
                        y.val[i*y.cols + j] += x1.val[k*x1.cols + i] * x2.val[j*x2.cols + k];
                    }
                }
            }
            return;
        }
    };
    struct Concat {

        static Mat col(const Mat &x1, const Mat &x2)
        {
            assert(x1.rows == x2.rows);
            Mat y(x1.rows, x1.cols + x2.cols);
            for (std::size_t i = 0; i < y.rows; i++) {
                for (std::size_t j = 0; i < y.cols; j++) {
                    if (j < x1.cols) {
                        y(i, j) = x1(i, j);
                    } else {
                        y(i, j) = x2(i, j - x1.cols);
                    }
                }
            }
            return y;
        }

        static Mat row(const Mat &x1, const Mat &x2)
        {
            assert(x1.cols == x2.cols);
            Mat y(x1.rows + x2.rows, x1.cols);
            for (std::size_t i = 0; i < y.rows; i++) {
                for (std::size_t j = 0; i < y.cols; j++) {
                    if (i < x1.rows) {
                        y(i, j) = x1(i, j);
                    } else {
                        y(i, j) = x2(i - x1.rows, j);
                    }
                }
            }
            return y;
        }
    };
    static void parse(std::istringstream &stream, std::size_t cols, Mat &x)
    {
        /* csv:<rows><cols><data> */
        std::size_t row = 0;
        std::size_t col = 0;
        /* rows */
        std::string rowData;
        std::getline(stream, rowData, ',');
        row = std::atoi(rowData.c_str());
        /* cols */
        std::string colData;
        std::getline(stream, colData, ',');
        col = std::atoi(colData.c_str());
        /* data */
        x = Mat(row, col);
        for (std::size_t i = 0; i < x.totalSize; i++) {
            std::string data;
            std::getline(stream, data, ',');
            x.val[i] = std::atof(data.c_str());
        }
        return;
    }

    static void toString(const Mat &x, std::string &line)
    {
        /* csv:<rows><cols><data> */
        line += std::to_string(x.rows) + "," + std::to_string(x.cols) + ",";
        for (std::size_t i = 0; i < x.rows; i++) {
            for (std::size_t j = 0; j < x.cols; j++) {
                line += std::to_string(x(i, j));
                if (j < x.cols - 1) {
                    line += ",";
                }
            }
        }
        return;
    }

};

}
#endif // MAT_HPP
