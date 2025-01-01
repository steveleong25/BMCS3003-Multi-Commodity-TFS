#ifndef OMP_FLOW_ALGORITHM_HPP
#define OMP_FLOW_ALGORITHM_HPP

#include <vector> 
#include "NetworkGraph.hpp"
#include "Commodity.hpp"

double parallel_calculate_path_weight(const Graph& g, const std::vector<int>& path);

void OMP_flowDistributionAlgorithm(Graph& g, vector<Commodity>& commodities, int num_of_iter);

vector<boost::graph_traits<Graph>::edge_descriptor> OMP_get_edges_with_flow(Graph& g);

#endif