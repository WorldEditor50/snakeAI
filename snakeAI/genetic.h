#ifndef GENETIC_H
#define GENETIC_H
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#define OPT_MAX 0
#define OPT_MIN 1
class Factor
{
public:
    Factor():fitness(0), relativeFitness(0), accumulateFitness(0){}
    ~Factor(){}
    Factor(std::size_t codeLen);
    Factor(const Factor& factor);
    void mutate();
    void crossover(Factor& factor);
    float decode();
    void copyTo(Factor& dstFactor);
    void swap(Factor& factor);
public:
    float fitness;
    float relativeFitness;
    float accumulateFitness;
    float maxValue;
    std::vector<char> code;
};
class Genetic
{
public:
    float run(char OptimizeType, std::size_t iterateNum);
    void show(int index);
    Genetic():crossRate(0.6), mutateRate(0.3){}
    ~Genetic(){}
    Genetic(float crossRate, float mutateRate, int maxGroupSize, int maxCodeLen);
    void create(float crossRate, float mutateRate, std::size_t maxGroupSize, int maxCodeLen);
    float objectFunction(float x);
private:
    int select();
    void crossover(int index1, int index2);
    void mutate(int index);
    void eliminate(char OptimizeType);
    void calculateFitness();
private:
    float crossRate;
    float mutateRate;
    Factor optimalFactor;
    std::vector<Factor> group;
};
#endif // GENETIC_H
