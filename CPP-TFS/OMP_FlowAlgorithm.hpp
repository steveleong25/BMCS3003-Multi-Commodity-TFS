#ifndef OMP_FLOW_ALGORITHM_HPP
#define OMP_FLOW_ALGORITHM_HPP

#include <vector> // Include the vector header
#include "NetworkGraph.hpp"
#include "Commodity.hpp"

double parallel_calculate_bottleneck(Graph& g);

bool parallel_isFlowExceedingCapacity(Graph& g);

void parallel_normalize_flows(Graph& g, double bottleneck_value);

void parallel_updateCommoditiesSent(std::vector<Commodity>& commodities, double bottleneck_value);

void parallel_recalculate_weights(Graph& g, double alpha);

double OMP_flowDistributionAlgorithm(Graph& g, std::vector<Commodity>& commodities, double epsilon, double alpha);

#endif#pragma once
