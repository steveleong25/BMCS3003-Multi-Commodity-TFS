#ifndef PROP_FLOW_ALGORITHM_HPP
#define PROP_FLOW_ALGORITHM_HPP

#include "NetworkGraph.hpp"

void sendFlow(NetworkGraph& graph, const std::vector<std::string>& path, double amount);

void equalDistributionAlgorithm(NetworkGraph& graph, std::vector<std::pair<std::string, std::string>> commodities, std::vector<double> demands);

void redistributeFlowForEqualization(NetworkGraph& graph, std::vector<std::pair<std::string, std::string>>& commodities, std::vector<double>& demands, std::vector<double>& flowDelivered, std::vector<double>& successRates);

#endif
