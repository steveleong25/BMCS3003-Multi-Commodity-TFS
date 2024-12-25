#ifndef PROP_FLOW_ALGORITHM_HPP
#define PROP_FLOW_ALGORITHM_HPP

#include <vector> // Include the vector header
#include "NetworkGraph.hpp"
#include "Commodity.hpp"

double calculate_bottleneck(Graph& g);

bool isFlowExceedingCapacity(Graph& g);

void normalize_flows(Graph& g, double bottleneck_value);

void updateCommoditiesSent(std::vector<Commodity>& commodities, double bottleneck_value);

void recalculate_weights(Graph& g, double alpha);

double flowDistributionAlgorithm(Graph& g, std::vector<Commodity>& commodities, double epsilon, double alpha);

#endif