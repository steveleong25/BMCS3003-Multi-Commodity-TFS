#ifndef MPI_PROP_FLOW_ALGORITHM_HPP
#define MPI_PROP_FLOW_ALGORITHM_HPP

#include <vector>
#include "NetworkGraph.hpp"

void MPI_redistributeFlowForEqualization(NetworkGraph& graph, vector<pair<string, string>>& commodities, vector<double>& demands, vector<double>& unitsDelivered, vector<double>& successRates);

void MPI_equalDistributionAlgorithm(NetworkGraph& graph, vector<pair<string, string>> commodities, vector<double> demands);

#endif