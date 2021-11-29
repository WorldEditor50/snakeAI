#include "mat.h"


matrix::Mat matrix::Mat::ZERO(std::size_t rows, std::size_t cols)
{
    return Mat(rows, cols);
}

matrix::Mat matrix::Mat::IDENTITY(std::size_t rows, std::size_t cols)
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < y.rows; i++) {
        for (std::size_t j = 0; j < y.cols; j++) {
            if (i == j) {
                y.data[i][j] = 1;
            }
        }
    }
    return y;
}

matrix::Mat &matrix::Mat::operator=(const Mat &r)
{
    if (shapeEqual(r) == false) {
        std::cout<<"Mat: operator= : different shape"
                <<" rows:"<<rows<<" r.rows:"<<r.rows
                <<" cols:"<<cols<<" r.cols:"<<r.cols;
        return *this;
    }
    if (this == &r) {
        return *this;
    }
    for (std::size_t i = 0; i < data.size(); i++) {
        for (std::size_t j = 0; j < data[0].size(); j++) {
            data[i][j] = r.data[i][j];
        }
    }
    return *this;
}

matrix::Mat matrix::Mat::operator *(const Mat &x) const
{
    if (cols != x.rows) {
        return Mat();
    }
    Mat y(rows, x.cols);
    for (std::size_t i = 0; i < y.rows; i++) {
        for (std::size_t j = 0; j < y.cols; j++) {
            for (std::size_t k = 0; k < cols; k++) {
                y.data[i][j] += data[i][k] * x.data[k][j];
            }
        }
    }
    return y;
}

matrix::Mat matrix::Mat::operator /(const Mat &x) const
{
    if (shapeEqual(x) == false) {
        std::cout<<"Mat: operator/ : different shape"
                <<" rows:"<<rows<<" x.rows:"<<x.rows
                <<" cols:"<<cols<<" x.cols:"<<x.cols;
        return *this;
    }
    Mat y(rows, cols);
    for (std::size_t i = 0; i < y.rows; i++) {
        for (std::size_t j = 0; j < y.cols; j++) {
            y.data[i][j] = data[i][j] / x.data[i][j];
        }
    }
    return y;
}

matrix::Mat matrix::Mat::operator +(const Mat &x) const
{
    if (shapeEqual(x) == false) {
        std::cout<<"Mat: operator+ : different shape"
                <<" rows:"<<rows<<" x.rows:"<<x.rows
                <<" cols:"<<cols<<" x.cols:"<<x.cols;
        return *this;
    }
    Mat y(rows, cols);
    for (std::size_t i = 0; i < y.rows; i++) {
        for (std::size_t j = 0; j < y.cols; j++) {
            y.data[i][j] = data[i][j] + x.data[i][j];
        }
    }
    return y;
}

matrix::Mat matrix::Mat::operator -(const Mat &x) const
{
    if (shapeEqual(x) == false) {
        std::cout<<"Mat: operator- : different shape"
                <<" rows:"<<rows<<" x.rows:"<<x.rows
                <<" cols:"<<cols<<" x.cols:"<<x.cols;
        return *this;
    }
    Mat y(rows, cols);
    for (std::size_t i = 0; i < y.rows; i++) {
        for (std::size_t j = 0; j < y.cols; j++) {
            y.data[i][j] = data[i][j] - x.data[i][j];
        }
    }
    return y;
}

matrix::Mat matrix::Mat::operator %(const Mat &x) const
{
    if (shapeEqual(x) == false) {
        std::cout<<"Mat: operator% : different shape"
                <<" rows:"<<rows<<" x.rows:"<<x.rows
                <<" cols:"<<cols<<" x.cols:"<<x.cols;
        return *this;
    }
    Mat y(rows, cols);
    for (std::size_t i = 0; i < y.rows; i++) {
        for (std::size_t j = 0; j < y.cols; j++) {
            y.data[i][j] = data[i][j] * x.data[i][j];
        }
    }
    return y;
}

matrix::Mat &matrix::Mat::operator /=(const Mat &x)
{
    if (shapeEqual(x) == false) {
        std::cout<<"Mat: operator/= : different shape"
                <<" rows:"<<rows<<" x.rows:"<<x.rows
                <<" cols:"<<cols<<" x.cols:"<<x.cols;
        return *this;
    }
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            data[i][j] /= x.data[i][j];
        }
    }
    return *this;
}

matrix::Mat &matrix::Mat::operator +=(const Mat &x)
{
    if (shapeEqual(x) == false) {
        std::cout<<"Mat: operator+= : different shape"
                <<" rows:"<<rows<<" x.rows:"<<x.rows
                <<" cols:"<<cols<<" x.cols:"<<x.cols;
        return *this;
    }
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            data[i][j] += x.data[i][j];
        }
    }
    return *this;
}

matrix::Mat &matrix::Mat::operator -=(const Mat &x)
{
    if (shapeEqual(x) == false) {
        std::cout<<"Mat: operator-= : different shape"
                <<" rows:"<<rows<<" x.rows:"<<x.rows
                <<" cols:"<<cols<<" x.cols:"<<x.cols;
        return *this;
    }
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            data[i][j] -= x.data[i][j];
        }
    }
    return *this;
}

matrix::Mat &matrix::Mat::operator %=(const Mat &x)
{
    if (shapeEqual(x) == false) {
        std::cout<<"Mat: operator%= : different shape"
                <<" rows:"<<rows<<" x.rows:"<<x.rows
                <<" cols:"<<cols<<" x.cols:"<<x.cols;
        return *this;
    }
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            data[i][j] *= x.data[i][j];
        }
    }
    return *this;
}

matrix::Mat matrix::Mat::operator *(T x) const
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < y.rows; i++) {
        for (std::size_t j = 0; j < y.cols; j++) {
            y.data[i][j] = data[i][j] * x;
        }
    }
    return y;
}

matrix::Mat matrix::Mat::operator /(T x) const
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < y.rows; i++) {
        for (std::size_t j = 0; j < y.cols; j++) {
            y.data[i][j] = data[i][j] / x;
        }
    }
    return y;
}

matrix::Mat matrix::Mat::operator +(T x) const
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < y.rows; i++) {
        for (std::size_t j = 0; j < y.cols; j++) {
            y.data[i][j] = data[i][j] + x;
        }
    }
    return y;
}

matrix::Mat matrix::Mat::operator -(T x) const
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < y.rows; i++) {
        for (std::size_t j = 0; j < y.cols; j++) {
            y.data[i][j] = data[i][j] - x;
        }
    }
    return y;
}

matrix::Mat &matrix::Mat::operator *=(T x)
{
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            data[i][j] *= x;
        }
    }
    return *this;
}

matrix::Mat &matrix::Mat::operator /=(T x)
{
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            data[i][j] /= x;
        }
    }
    return *this;
}

matrix::Mat &matrix::Mat::operator +=(T x)
{
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            data[i][j] += x;
        }
    }
    return *this;
}

matrix::Mat &matrix::Mat::operator -=(T x)
{
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            data[i][j] -= x;
        }
    }
    return *this;
}

matrix::Mat matrix::Mat::tr() const
{
    Mat y(cols, rows);
    for (std::size_t i = 0; i < y.rows; i++) {
        for (std::size_t j = 0; j < y.cols; j++) {
            y.data[i][j] = data[j][i];
        }
    }
    return y;
}

matrix::Mat matrix::Mat::sub(std::size_t pr, std::size_t pc, std::size_t sr, std::size_t sc) const
{
    Mat y(sr, sc);
    for (std::size_t i = 0; i < y.rows; i++) {
        for (std::size_t j = 0; j < y.cols; j++) {
            y.data[i][j] = data[i + pr][j + pc];
        }
    }
    return y;
}

matrix::Mat matrix::Mat::flatten() const
{
    Mat y(rows * cols, 1);
    std::size_t k = 0;
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            y.data[k][0] = data[i][j];
            k++;
        }
    }
    return y;
}

matrix::Mat matrix::Mat::f(std::function<double (double)> func) const
{
    Mat y(rows, cols);
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            y.data[i][j] = func(data[i][j]);
        }
    }
    return y;
}

void matrix::Mat::set(std::size_t pr, std::size_t pc, const matrix::Mat &x)
{
    for (std::size_t i = pr; i < pr + x.rows; i++) {
        for (std::size_t j = pc; j < pc + x.cols; j++) {
            data[i][j] = x.data[i - pr][j - pc];
        }
    }
    return;
}

void matrix::Mat::zero()
{
    for (std::size_t i = 0; i < data.size(); i++) {
        for (std::size_t j = 0; j < data[0].size(); j++) {
            data[i][j] = 0;
        }
    }
    return;
}

void matrix::Mat::full(T x)
{
    for (std::size_t i = 0; i < data.size(); i++) {
        for (std::size_t j = 0; j < data[0].size(); j++) {
            data[i][j] = x;
        }
    }
    return;
}

void matrix::Mat::EMA(const Mat &r, T rho)
{
    if (shapeEqual(r) == false) {
        std::cout<<"Mat: EMA : different shape"
                <<" rows:"<<rows<<" r.rows:"<<r.rows
                <<" cols:"<<cols<<" r.cols:"<<r.cols;
        return;
    }
    for (std::size_t i = 0; i < data.size(); i++) {
        for (std::size_t j = 0; j < data[0].size(); j++) {
            data[i][j] = (1 - rho) * data[i][j] + rho * r.data[i][j];
        }
    }
    return;
}

void matrix::Mat::uniformRand(std::default_random_engine &engine)
{
    std::uniform_real_distribution<T> uniform(-1, 1);
    for (std::size_t i = 0; i < data.size(); i++) {
        for (std::size_t j = 0; j < data[0].size(); j++) {
            data[i][j] = uniform(engine);
        }
    }
    return;
}

void matrix::Mat::show() const
{
    for (std::size_t i = 0; i < data.size(); i++) {
        for (std::size_t j = 0; j < data[0].size(); j++) {
            std::cout<<data[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
    return;
}

matrix::Mat matrix::kronecker(const matrix::Mat &x1, const matrix::Mat &x2)
{
    Mat y(x1.rows * x2.rows, x1.cols * x2.cols);
    for (std::size_t i = 0; i < x1.rows; i++) {
        for (std::size_t j = 0; j < x1.cols; j++) {
            for (std::size_t k = 0; k < x2.rows; k++) {
                for (std::size_t l = 0; l < x2.cols; l++) {
                    y.data[i*x2.rows + k][j*x2.cols + l] = x1.data[i][j] * x2.data[k][l];
                }
            }
        }
    }
    return y;
}

matrix::Mat matrix::SQRT(const matrix::Mat &x)
{
    Mat y(x.shape());
    for (std::size_t i = 0; i < x.rows; i++) {
        for (std::size_t j = 0; j < x.cols; j++) {
            y.data[i][j] = sqrt(x.data[i][j]);
        }
    }
    return y;
}

matrix::Mat matrix::EXP(const matrix::Mat &x)
{
    Mat y(x.shape());
    for (std::size_t i = 0; i < x.rows; i++) {
        for (std::size_t j = 0; j < x.cols; j++) {
            y.data[i][j] = exp(x.data[i][j]);
        }
    }
    return y;
}

matrix::Mat matrix::SIN(const matrix::Mat &x)
{
    Mat y(x.shape());
    for (std::size_t i = 0; i < x.rows; i++) {
        for (std::size_t j = 0; j < x.cols; j++) {
            y.data[i][j] = sin(x.data[i][j]);
        }
    }
    return y;
}

matrix::Mat matrix::COS(const matrix::Mat &x)
{
    Mat y(x.shape());
    for (std::size_t i = 0; i < x.rows; i++) {
        for (std::size_t j = 0; j < x.cols; j++) {
            y.data[i][j] = cos(x.data[i][j]);
        }
    }
    return y;
}

matrix::Mat matrix::TANH(const matrix::Mat &x)
{
    Mat y(x.shape());
    for (std::size_t i = 0; i < x.rows; i++) {
        for (std::size_t j = 0; j < x.cols; j++) {
            y.data[i][j] = tanh(x.data[i][j]);
        }
    }
    return y;
}

void matrix::test()
{
    std::default_random_engine engine;
    std::cout<<"x1:"<<std::endl;
    Mat x1(3, 2);
    x1.uniformRand(engine);
    x1.show();
    std::cout<<"x2:"<<std::endl;
    Mat x2(2, 4);
    x2.uniformRand(engine);
    x2.show();
    std::cout<<"y:"<<std::endl;
    Mat y = x1 * x2;
    y.show();
    Mat yt = y.tr();
    yt.show();
    return;
}
