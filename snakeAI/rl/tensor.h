#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
using T = double;

template<std::size_t N>
struct Tensor_
{
    using Type = std::vector<typename Tensor_<N - 1>::Type>;
};

template<>
struct Tensor_<1>
{
    using Type = std::vector<T>;
};

template<>
struct Tensor_<0>
{
    using Type = T;
};

class Tensor
{
public:
   std::size_t size_;
   std::vector<std::size_t> shape;
   std::vector<T> data;
public:
   explicit Tensor(const std::vector<std::size_t> &shape_);
   void zero();
   void fill(T value);
   template<typename ...Dim>
   T &at(Dim ...dim)
   {
       std::vector<std::size_t> dims{dim...};
       std::size_t pos = 0;
       for (std::size_t i = 0; i < shape.size() - 1; i++) {
           pos += shape[i] * dims[i];
       }
       pos += dims[shape.size() - 1];
       return data[pos];
   }
   Tensor operator + (const Tensor &x) const;
   Tensor operator - (const Tensor &x) const;
   Tensor operator * (const Tensor &x) const;
   Tensor operator / (const Tensor &x) const;
   Tensor &operator += (const Tensor &x);
   Tensor &operator -= (const Tensor &x);
   Tensor &operator *= (const Tensor &x);
   Tensor &operator /= (const Tensor &x);
   Tensor operator + (T x) const;
   Tensor operator - (T x) const;
   Tensor operator * (T x) const;
   Tensor operator / (T x) const;
   Tensor &operator += (T x);
   Tensor &operator -= (T x);
   Tensor &operator *= (T x);
   Tensor &operator /= (T x);
};
Tensor kronecter(const Tensor &x1, const Tensor &x2);
#endif // TENSOR_H
