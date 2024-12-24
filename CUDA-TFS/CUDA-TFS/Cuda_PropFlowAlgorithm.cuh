#ifndef CUDA_PROP_FLOW_ALGORITHM_H
#define CUDA_PROP_FLOW_ALGORITHM_H

#include <vector>
#include "NetworkGraph.hpp"
#include "Commodity.hpp"
#include <cuda_runtime.h>

// Declare CUDA-compatible functions
__global__ void calculate_bottleneck_kernel(const int* edge_usage, const int* capacities, int num_edges, int* bottleneck_results);
__global__ void normalize_flows_kernel(int* flows, const int* capacities, int num_edges, double bottleneck_value);
__global__ void recalculate_weights_kernel(double* weights, const int* flows, double alpha, int num_edges);

// Host-side function to orchestrate the CUDA flow distribution algorithm
double cudaFlowDistributionAlgorithm(Graph& g, std::vector<Commodity>& commodities, double epsilon, double alpha);

#endif // CUDA_PROP_FLOW_ALGORITHM_H
