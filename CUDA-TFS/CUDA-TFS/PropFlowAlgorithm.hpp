#ifndef PROP_FLOW_ALGORITHM_HPP
#define PROP_FLOW_ALGORITHM_HPP

#include <vector> // Include the vector header
#include <string> // Include the string header for std::string
#include "NetworkGraph.hpp"
#include "Commodity.hpp"

// Using std::vector and std::pair explicitly or ensure it's within the std namespace
//void sendFlow(NetworkGraph& graph, const std::vector<std::string>& path, double amount);
//
//void equalDistributionAlgorithm(NetworkGraph& graph, std::vector<std::pair<std::string, std::string>> commodities, std::vector<double> demands);
//
//void redistributeFlowForEqualization(NetworkGraph& graph, std::vector<std::pair<std::string, std::string>>& commodities, std::vector<double>& demands, std::vector<double>& flowDelivered, std::vector<double>& successRates);
//
//void OMP_redistributeFlowForEqualization(NetworkGraph& graph, std::vector<std::pair<std::string, std::string>>& commodities, std::vector<double>& demands, std::vector<double>& unitsDelivered, std::vector<double>& successRates);
//
//void OMP_equalDistributionAlgorithm(NetworkGraph& graph, std::vector<std::pair<std::string, std::string>> commodities, std::vector<double> demands);

int calculate_bottleneck(const Graph& g, const std::vector<std::vector<int>>& assigned_paths);

bool isFlowExceedingCapacity(Graph& g);

void normalize_flows(Graph& g, std::vector<std::vector<int>>& assigned_paths, double bottleneck_value);

void updateCommoditiesSent(vector<Commodity>& commodities, double bottleneck_value);

void recalculate_weights(Graph& g, double alpha);

double flowDistributionAlgorithm(Graph& g, std::vector<Commodity>& commodities, double epsilon, double alpha);

double OMP_flowDistributionAlgorithm(Graph& g, std::vector<Commodity>& commodities, double epsilon, double alpha);

#endif