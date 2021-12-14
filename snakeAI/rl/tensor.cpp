#include "tensor.h"

Tensor::Tensor(const std::vector<std::size_t> &shape_)
    :size_(1), shape(shape_)
{
    for (std::size_t i = 0; i < shape_.size(); i++) {
        size_ *= shape_[i];
    }
    data = std::vector<T>(size_, 0);
}

void Tensor::zero()
{
    data.assign(size_, 0);
    return;
}

void Tensor::fill(T value)
{
    data.assign(size_, value);
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
