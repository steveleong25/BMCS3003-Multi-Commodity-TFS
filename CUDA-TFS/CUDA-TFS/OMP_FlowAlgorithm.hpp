#ifndef OMP_FLOW_ALGORITHM_HPP
#define OMP_FLOW_ALGORITHM_HPP

#include <vector> 
#include "NetworkGraph.hpp"
#include "Commodity.hpp"

double parallel_calculate_bottleneck(Graph& g, vector<boost::graph_traits<Graph>::edge_descriptor>& edges_with_flow);

bool parallel_isFlowExceedingCapacity(Graph& g, vector<boost::graph_traits<Graph>::edge_descriptor>& edges_with_flow);

void parallel_normalize_flows(Graph& g, double bottleneck_value, vector<boost::graph_traits<Graph>::edge_descriptor>& edges_with_flow);

void parallel_updateCommoditiesSent(std::vector<Commodity>& commodities, double bottleneck_value);

void parallel_recalculate_weights(Graph& g, double alpha, vector<boost::graph_traits<Graph>::edge_descriptor>& edges_with_flow);

double OMP_flowDistributionAlgorithm(Graph& g, std::vector<Commodity>& commodities, double epsilon, double alpha);

vector<boost::graph_traits<Graph>::edge_descriptor> OMP_get_edges_with_flow(Graph& g);

#endif