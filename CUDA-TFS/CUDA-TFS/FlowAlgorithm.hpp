#ifndef FLOW_ALGORITHM_HPP
#define FLOW_ALGORITHM_HPP

#include <vector> 
#include "NetworkGraph.hpp"
#include "Commodity.hpp"

double calculate_path_weight(const Graph& g, const std::vector<int>& path);

void flowDistributionAlgorithm(Graph& g, vector<Commodity>& commodities, int num_of_iter);

vector<boost::graph_traits<Graph>::edge_descriptor> get_edges_with_flow(Graph& g);

#endif