#ifndef MPI_FLOW_ALGORITHM_HPP
#define MPI_FLOW_ALGORITHM_HPP

#include <vector> 
#include "NetworkGraph.hpp"
#include "Commodity.hpp"

//double calculate_bottleneck(Graph& g, vector<boost::graph_traits<Graph>::edge_descriptor> edges_with_flow);

//bool isFlowExceedingCapacity(Graph& g, vector<boost::graph_traits<Graph>::edge_descriptor> edges_with_flow);

//void normalize_flows(Graph& g, double bottleneck_value, vector<boost::graph_traits<Graph>::edge_descriptor> edges_with_flow);

//void updateCommoditiesSent(std::vector<Commodity>& commodities, double bottleneck_value);

//void recalculate_weights(Graph& g, double alpha, vector<boost::graph_traits<Graph>::edge_descriptor> edges_with_flow);

double MPI_flowDistributionAlgorithm(Graph& g, std::vector<Commodity>& commodities, double epsilon, double alpha);

//vector<boost::graph_traits<Graph>::edge_descriptor> get_edges_with_flow(Graph& g);

#endif