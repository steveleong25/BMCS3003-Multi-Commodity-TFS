#ifndef CUDA_FLOW_ALGORITHM_HPP
#define CUDA_FLOW_ALGORITHM_HPP

#include <vector> 
#include "NetworkGraph.hpp"
#include "Commodity.hpp"
#include <cuda_runtime.h>


__global__ void calculate_path_weights_kernel(const int* path, const double* edge_weights, int path_length, double* result);

double cuda_calculate_path_weight(const Graph& g, const std::vector<int>& path);

void CUDA_flowDistributionAlgorithm(Graph& g, vector<Commodity>& commodities, int num_of_iter);

vector<boost::graph_traits<Graph>::edge_descriptor> CUDA_get_edges_with_flow(Graph& g);

#endif