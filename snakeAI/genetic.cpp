#include "genetic.h"

double Genetic::run(char OptimizeType, std::size_t iterate)
{
    for (std::size_t i = 0; i < iterate; i++) {
        calculateFitness();
        int index1 = select();
        int index2 = select();
        if (index1 != index2) {
            crossover(index1, index2);
        }
        mutate(index1);
        mutate(index2);
        eliminate(OptimizeType);
	}
    return group[0].decode();
}

void Genetic::eliminate(char OptimizeType)
{
	/* sort */
    for (std::size_t i = 0; i < group.size() - 1; i++) {
        for (std::size_t j = i + 1; j < group.size(); j++) {
            if (OptimizeType == OPT_MAX) {
                if (group[i].fitness < group[j].fitness) {
                    group[i].swap(group[j]);
                }
            } else {
                if (group[i].fitness > group[j].fitness) {
                    group[i].swap(group[j]);
                }
            }

		}
	}
	/* eliminate */
    std::size_t len = group.size();
    int k = rand() % (len / 3);
    int j = 0;
    for (std::size_t i = len -1; i >= len - k; i--) {
        group[j].copyTo(group[i]);
        j++;
	}
	/* Show */
    this->show(0);
	return;
}

int Genetic::select()
{
    int index = 0;
    double p = double(rand() % 10000) / 10000;
	if (p <= group[0].accumulateFitness) {
        index = 0;
	} else {
        for (std::size_t i = 1; i < group.size() - 1; i++) {
			if (p > group[i].accumulateFitness && p <= group[i + 1].accumulateFitness) {
                index = i + 1;
                break;
			}
		}
	}
    return index;
}

void Genetic::crossover(int index1, int index2)
{
    double p = double(rand() % 10000) / 10000;
    if (p > crossRate) {
        return;
    }
    group[index1].crossover(group[index2]);
	return;
}

void Genetic::mutate(int index)
{
    double p = double(rand() % 10000) / 10000;
    if (p > mutateRate) {
		return;
	}
    group[index].mutate();
	return;
}


void Genetic::calculateFitness()
{
    double sum = 0.0;
    double value = 0.0;
	/* calulate fitness */
    for (std::size_t i = 0; i < group.size(); i++) {
        value = group[i].decode();
        group[i].fitness = objectFunction(value);
		sum += group[i].fitness;
	}
	/* calulate relative fitness */
    for (std::size_t i = 0; i < group.size(); i++) {
		group[i].relativeFitness = group[i].fitness /sum;
	}
    /* calculate accumulate fitness */
	group[0].accumulateFitness = group[0].relativeFitness;
    for (std::size_t i = 1; i < group.size(); i++) {
		group[i].accumulateFitness = group[i - 1].accumulateFitness + group[i].relativeFitness;
	}
	return;
}

void Genetic::show(int index)
{
    double value = 0.0;
    double fitness = 0.0;
    value = group[index].decode();
    fitness = objectFunction(value);
	return;
}

Genetic::Genetic(double crossRate, double mutateRate, int maxGroupSize, int maxCodeLen)
{
    create(crossRate, mutateRate, maxGroupSize, maxCodeLen);
}

void Genetic::create(double crossRate, double mutateRate, std::size_t maxGroupSize, int maxCodeLen)
{
    this->crossRate = crossRate;
    this->mutateRate = mutateRate;
    for (std::size_t i = 0; i < maxGroupSize; i++) {
        Factor factor(maxCodeLen);
        group.push_back(factor);
    }
    return;
}

double Genetic::objectFunction(double x)
{
    return 100 - x * x;
}

Factor::Factor(std::size_t codeLen)
{
    code = std::vector<char>(codeLen, 0);
    for (std::size_t i = 0; i < codeLen; i++) {
        code[i] = rand() % 2;
    }
    fitness = 0.0;
    relativeFitness = 0.0;
    accumulateFitness = 0.0;
}

Factor::Factor(const Factor& factor)
{
    if (this == &factor) {
        return;
    }
    std::size_t codeLen = factor.code.size();
    this->code = std::vector<char>(codeLen, 0);
    for (std::size_t i = 0; i < codeLen; i++) {
        code[i] = factor.code[i];
    }
    this->fitness = factor.fitness;
    this->relativeFitness = factor.relativeFitness;
    this->accumulateFitness = factor.accumulateFitness;
}

double Factor::decode()
{
    double value = 0.0;
    std::size_t len = code.size();
    std::size_t mid =  code.size() / 2;
    for (std::size_t i = 0; i < len; i++) {
        value += double(code[i] * pow(2, mid - i - 1));
    }
    if (code[len - 1] == 1) {
        value *= -1;
    }
    return value;
}

void Factor::copyTo(Factor &dstFactor)
{
    dstFactor.code.assign(code.begin(), code.end());
    return;
}

void Factor::swap(Factor &factor)
{
    for (std::size_t i = 0; i < code.size(); i++) {
        char tmp = factor.code[i];
        factor.code[i] = code[i];
        code[i] = tmp;
    }
}

void Factor::mutate()
{
    int k = rand() % code.size();
    if (code[k] == 1) {
        code[k] = 0;
    } else {
        code[k] = 1;
    }
    return;
}

void Factor::crossover(Factor& factor)
{
    std::size_t crossNum = rand() % code.size();
    for (std::size_t i = 0; i < crossNum; i++) {
        int k = rand() % code.size();
        char bit = code[k];
        code[k] = factor.code[k];
        factor.code[k] = bit;
	}
    return;
}
