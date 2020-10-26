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
    Factor(int codeLen);
    Factor(const Factor& factor);
    void mutate();
    void crossover(Factor& factor);
    double decode();
    void copyTo(Factor& dstFactor);
    void swap(Factor& factor);
public:
    double fitness;
    double relativeFitness;
    double accumulateFitness;
    double maxValue;
    std::vector<char> code;
};
class Genetic
{
public:
    double run(char OptimizeType, int iterateNum);
    void show(int index);
    Genetic():crossRate(0.6), mutateRate(0.3){}
    ~Genetic(){}
    Genetic(double crossRate, double mutateRate, int maxGroupSize, int maxCodeLen);
    void create(double crossRate, double mutateRate, int maxGroupSize, int maxCodeLen);
    double objectFunction(double x);
private:
    int select();
    void crossover(int index1, int index2);
    void mutate(int index);
    void eliminate(char OptimizeType);
    void calculateFitness();
private:
    double crossRate;
    double mutateRate;
    Factor optimalFactor;
    std::vector<Factor> group;
};
#endif // GENETIC_H
