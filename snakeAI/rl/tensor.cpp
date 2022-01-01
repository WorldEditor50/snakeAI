#include "tensor.h"
std::default_random_engine Tensor::engine;

void Tensor::uniform(T inf, T sup)
{
    std::uniform_real_distribution<T> u(inf, sup);
    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] = u(Tensor::engine);
    }
    return;
}

void Tensor::show()
{
    if (size_ == 0) {
        std::cout<<"empty tensor"<<std::endl;
        return;
    }
    for (std::size_t i = 0; i < data.size(); i++) {
        std::cout<<data[i]<<" ";
    }
    std::cout<<std::endl;
    return;
}

Tensor Tensor::operator +(const Tensor &x) const
{
    Tensor y(shape);
    for (std::size_t i = 0; i < data.size(); i++) {
        y.data[i] = data[i] + x.data[i];
    }
    return y;
}

Tensor Tensor::operator -(const Tensor &x) const
{
    Tensor y(shape);
    for (std::size_t i = 0; i < data.size(); i++) {
        y.data[i] = data[i] - x.data[i];
    }
    return y;
}

Tensor Tensor::operator *(const Tensor &x) const
{
    Tensor y(shape);
    for (std::size_t i = 0; i < data.size(); i++) {
        y.data[i] = data[i] * x.data[i];
    }
    return y;
}

Tensor Tensor::operator /(const Tensor &x) const
{
    Tensor y(shape);
    for (std::size_t i = 0; i < data.size(); i++) {
        y.data[i] = data[i] / x.data[i];
    }
    return y;
}

Tensor &Tensor::operator +=(const Tensor &x)
{
    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] += x.data[i];
    }
    return *this;
}

Tensor &Tensor::operator -=(const Tensor &x)
{
    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] -= x.data[i];
    }
    return *this;
}

Tensor &Tensor::operator *=(const Tensor &x)
{
    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] *= x.data[i];
    }
    return *this;
}

Tensor &Tensor::operator /=(const Tensor &x)
{
    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] /= x.data[i];
    }
    return *this;
}

Tensor Tensor::operator +(T x) const
{
    Tensor y(shape);
    for (std::size_t i = 0; i < data.size(); i++) {
        y.data[i] = data[i] + x;
    }
    return y;
}

Tensor Tensor::operator -(T x) const
{
    Tensor y(shape);
    for (std::size_t i = 0; i < data.size(); i++) {
        y.data[i] = data[i] - x;
    }
    return y;
}

Tensor Tensor::operator *(T x) const
{
    Tensor y(shape);
    for (std::size_t i = 0; i < data.size(); i++) {
        y.data[i] = data[i] * x;
    }
    return y;
}

Tensor Tensor::operator /(T x) const
{
    Tensor y(shape);
    for (std::size_t i = 0; i < data.size(); i++) {
        y.data[i] = data[i] / x;
    }
    return y;
}

Tensor &Tensor::operator +=(T x)
{
    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] += x;
    }
    return *this;
}

Tensor &Tensor::operator -=(T x)
{
    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] -= x;
    }
    return *this;
}

Tensor &Tensor::operator *=(T x)
{
    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] *= x;
    }
    return *this;
}

Tensor &Tensor::operator /=(T x)
{
    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] /= x;
    }
    return *this;
}

void Tensor::test()
{
    Tensor x(2, 4, 5, 3);
    //x.uniform(0, 9);
    x.data = std::vector<T>{0, 1, 2,
                            3, 4, 5,
                            6, 7, 8,
                            5, 3, 2,
                            7, 9, 1,

                            1, 0, 1,
                            0, 1, 1,
                            0, 0, 1,
                            1, 1, 0,
                            1, 0, 0,

                            2, 0, 0,
                            0, 2, 0,
                            0, 0, 2,
                            0, 2, 0,
                            0, 0, 2,

                            1, 2, 4,
                            8, 16, 32,
                            64, 128, 256,
                            8, 16, 32,
                            64, 128, 256,


                            2, 2, 2,
                            3, 4, 5,
                            6, 7, 8,
                            5, 3, 2,
                            7, 9, 1,

                            1, 0, 1,
                            0, 1, 1,
                            0, 0, 1,
                            1, 1, 0,
                            1, 0, 0,

                            3, 0, 0,
                            0, 3, 0,
                            0, 0, 3,
                            0, 3, 0,
                            0, 0, 3,

                            1, 2, 4,
                            8, 16, 32,
                            64, 128, 256,
                            8, 16, 32,
                            2, 2, 2};
    for (std::size_t h = 0; h < x.dimension[0]; h++) {
        for (std::size_t k = 0; k < x.dimension[1]; k++) {
            for (std::size_t i = 0; i < x.dimension[2]; i++) {
                for (std::size_t j = 0; j < x.dimension[3]; j++) {
                    std::cout<<x.at(h, k, i, j)<<" ";
                    //std::cout<<x.data[h*60 + k*15 + i*3 +j]<<" ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }
    Tensor s = x.sub(1, 2);
    s.show();
    return;
}
