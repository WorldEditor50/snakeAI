#ifndef TENSOR_H
#define TENSOR_H
#include <iostream>
#include <vector>
#include <random>

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
   std::vector<std::size_t> dimension;
   std::vector<std::size_t> shape;
   std::vector<T> data;
   static std::default_random_engine engine;
public:
   Tensor():size_(0){}
   explicit Tensor(const std::vector<std::size_t> &dim):size_(1),dimension(dim)
   {
       shape = std::vector<std::size_t>(dimension.size(), 1);
       for (std::size_t i = 0; i < dimension.size(); i++) {
           size_ *= dimension[i];
       }
       for (std::size_t i = 0; i < dimension.size() - 1; i++) {
           for (std::size_t j = i + 1; j < dimension.size(); j++) {
                shape[i] *= dimension[j];
           }
       }
       data = std::vector<T>(size_, 0);
   }
   template<typename ...Dim>
   explicit Tensor(Dim ...dim):size_(1),dimension(std::vector<std::size_t>{dim...})
   {
       shape = std::vector<std::size_t>(dimension.size(), 1);
       for (std::size_t i = 0; i < dimension.size(); i++) {
           size_ *= dimension[i];
       }
       for (std::size_t i = 0; i < dimension.size() - 1; i++) {
           for (std::size_t j = i + 1; j < dimension.size(); j++) {
                shape[i] *= dimension[j];
           }
       }
       data = std::vector<T>(size_, 0);
   }

   Tensor(const Tensor &r)
       :size_(r.size_),dimension(r.dimension),shape(r.shape), data(r.data){}
   Tensor &operator=(const Tensor &r)
   {
       if (this == &r) {
           return *this;
       }
       size_ = r.size_;
       dimension = r.dimension;
       shape = r.shape;
       data = r.data;
       return *this;
   }

   template<typename ...Dim>
   inline T &at(Dim ...di)
   {
       std::size_t indexs[] = {di...};
       std::size_t pos = 0;
       for (std::size_t i = 0; i < shape.size(); i++) {
           pos += shape[i] * indexs[i];
       }
       return data[pos];
   }

   template<typename ...Dim>
   Tensor sub(Dim ...di)
   {
       Tensor y;
       std::vector<std::size_t> dims = std::vector<std::size_t>{di...};
       if (dims.size() >= dimension.size()) {
           return y;
       }
       for (std::size_t i = 0; i < dims.size(); i++) {
           if (dims[i] > dimension[i]) {
               return y;
           }
       }
       std::vector<std::size_t> subDims(dimension.begin() + dims.size(), dimension.end());
       y = Tensor(subDims);
       std::size_t pos = 0;
       for (std::size_t i = 0; i < dims.size(); i++) {
           pos += shape[i] * dims[i];
       }
       for (std::size_t i = 0; i < y.size_; i++) {
           y.data[i] = data[i + pos];
       }
       return y;
   }
   inline bool empty() const {return size_ == 0;}
   void zero(){data.assign(size_, 0);}
   void fill(T value){data.assign(size_, value);}
   inline T &operator[](std::size_t i) {return data[i];}
   void uniform(T inf, T sup);
   void show();
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
   static void test();
};
#endif // TENSOR_H
