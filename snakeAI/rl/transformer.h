#ifndef TRANSFORMER_H
#define TRANSFORMER_H
#include "util.h"

namespace RL {

class Embedding
{
public:

public:
    void forward(const Vec &x);
    void backward();
};

class PositionalEncoder
{
public:
    void forward(const Vec &x);
};

Vec attention(const Mat &Q, const Mat &K, const Mat &mask);

class ScaleDotProductAttention
{
public:

};

class MultiHeadAttention
{
public:
    std::size_t dim;
public:
    void forward(const Mat &Q, const Mat &K, const Mat &V, const Mat &mask);
};

class PositionWiseFeedForward
{
public:
    void forward(const Vec &x);
};

class Norm
{
public:
    void forward(const Vec &x);
};

class SubLayerConnection
{
public:
    void forward(const Vec &x);
};

class Encoder
{
public:

};

class Decoder
{
public:

};

class Transformer
{
public:
    std::vector<Encoder> encoders;
    std::vector<Decoder> decoders;
public:
    Transformer();
    void forward(const Vec &x);
};

}
#endif // TRANSFORMER_H
