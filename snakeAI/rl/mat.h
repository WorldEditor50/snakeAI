#ifndef MAT_H
#define MAT_H
#include <vector>
#include <random>
#include <functional>
#include <cmath>
#include <ctime>
#include <iostream>
namespace matrix {

using T = double;
class Mat
{
public:
    std::size_t rows;
    std::size_t cols;
    std::vector<std::vector<T> > data;
    class Size
    {
    public:
        std::size_t r;
        std::size_t c;
    public:
        Size():r(0),c(0){}
        Size(std::size_t r_, std::size_t c_):r(r_), c(c_){}
        bool operator == (const Size &s) const
        {
            return r == s.r && c == s.c;
        }
    };

public:
    Mat(){}
    Mat(const Size &s)
        :rows(s.r),cols(s.c),data(s.r, std::vector<T>(s.c, 0)){}
    Mat(std::size_t rows_, std::size_t cols_)
        :rows(rows_),cols(cols_),data(rows_, std::vector<T>(cols_, 0)){}
    Mat(std::size_t rows_, std::size_t cols_, T value)
        :rows(rows_),cols(cols_),data(rows_, std::vector<T>(cols_, value)){}
    Mat(const Mat &r)
    :rows(r.rows),cols(r.cols),data(r.data){}
    static Mat ZERO(std::size_t rows, std::size_t cols);
    static Mat IDENTITY(std::size_t rows, std::size_t cols);
    inline Size shape() const {return Size(rows, cols);}
    inline bool shapeEqual(const Mat &r) const {return rows == r.rows && cols == r.cols;}
    inline std::vector<T> &operator[](std::size_t i){return data[i];}
    Mat &operator=(const Mat &r);
    /* object operation */
    Mat operator * (const Mat &x) const;
    Mat operator / (const Mat &x) const;
    Mat operator + (const Mat &x) const;
    Mat operator - (const Mat &x) const;
    Mat operator % (const Mat &x) const;
    Mat &operator /= (const Mat &x);
    Mat &operator += (const Mat &x);
    Mat &operator -= (const Mat &x);
    Mat &operator %= (const Mat &x);
    Mat operator * (T x) const;
    Mat operator / (T x) const;
    Mat operator + (T x) const;
    Mat operator - (T x) const;
    Mat &operator *= (T x);
    Mat &operator /= (T x);
    Mat &operator += (T x);
    Mat &operator -= (T x);
    Mat tr() const;
    Mat sub(std::size_t sr, std::size_t sc, std::size_t offsetr, std::size_t offsetc) const;
    Mat flatten() const;
    Mat f(std::function<double(double)> func) const;
    void set(std::size_t sr, std::size_t sc, const Mat &x);
    void zero();
    void full(T x);
    void EMA(const Mat &r, T rho);
    void uniformRand(std::default_random_engine &engine);
    void show() const;
    /* static operation */
    static void mul(const Mat &x1, const Mat &x2, Mat &y);
    static void dot(const Mat &x1, const Mat &x2, Mat &y);
    static void div(const Mat &x1, const Mat &x2, Mat &y);
    static void add(const Mat &x1, const Mat &x2, Mat &y);
    static void minus(const Mat &x1, const Mat &x2, Mat &y);
};
std::size_t argmax(const Mat &x);
std::size_t argmin(const Mat &x);
T max(const Mat &x);
T min(const Mat &x);
T sum(const Mat &x);
T mean(const Mat &x);
T var(const Mat &x);
T var(const Mat &x, T u);
T cov(const Mat &x, const Mat &y);
T cov(const Mat &x, T ux, const Mat &y, T uy);
Mat kronecker(const Mat &x1, const Mat &x2);
Mat SQRT(const Mat &x);
Mat EXP(const Mat &x);
Mat TANH(const Mat &x);
Mat SIN(const Mat &x);
Mat COS(const Mat &x);
struct Tanh {
    inline static T dtanh(T y) {return 1 - y*y;}
    inline static Mat _(const Mat &x) {return TANH(x);}
    inline static Mat d(const Mat &y) {return y.f(dtanh);}
};

struct Relu {
    inline static T relu(T x) {return x > 0 ? x : 0;}
    inline static T drelu(T y) {return y > 0 ? 1 : 0;}
    inline static Mat _(const Mat &x) {return x.f(relu);}
    inline static Mat d(const Mat &y) {return y.f(drelu);}
};

struct LeakyRelu {
    inline static T leakyRelu(T x) {return x > 0 ? x : 0.01*x;}
    inline static T dLeakyRelu(T y) {return y > 0 ? 1 : 0.01;}
    inline static Mat _(const Mat &x) {return x.f(leakyRelu);}
    inline static Mat d(const Mat &y) {return y.f(dLeakyRelu);}
};

struct Linear {
    inline static Mat _(const Mat & x) {return x;}
    inline static Mat d(const Mat &y) {return Mat(y.rows, y.cols, 1);}
};
void test();
}
#endif // MAT_H
