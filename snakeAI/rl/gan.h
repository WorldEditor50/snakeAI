#ifndef GAN_H
#define GAN_H
#include "bpnn.h"
namespace RL {

class GAN
{
public:
    GAN(){}
    GAN(std::size_t generatorInputDim,
        std::size_t generatorOutputDim,
        std::size_t discriminatorInputDim,
        std::size_t discriminatorOutputDim);
protected:
    BPNN generator;
    BPNN discriminator;
};
}
#endif // GAN_H
