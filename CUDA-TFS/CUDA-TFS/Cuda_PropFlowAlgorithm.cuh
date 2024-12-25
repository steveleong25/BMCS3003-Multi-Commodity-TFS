#ifndef CUDA_PROP_FLOW_ALGORITHM_H
#define CUDA_PROP_FLOW_ALGORITHM_H

#include "NetworkGraph.hpp"
#include "Commodity.hpp"
#include <cuda_runtime.h>

__global__ void calculateBottleneckKernel(const EdgeProperties* edges, int numEdges, double* bottleneckValues);
__global__ void normalizeFlowsKernel(EdgeProperties* edges, int numEdges, double bottleneckValue);
__global__ void recalculateWeightsKernel(EdgeProperties* edges, int numEdges, double alpha);

double CUDA_flowDistributionAlgorithm(Graph& g, std::vector<Commodity>& commodities, double epsilon, double alpha);

#endif
