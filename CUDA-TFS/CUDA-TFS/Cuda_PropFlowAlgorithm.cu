#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <cuda_runtime.h>
#include "NetworkGraph.hpp"
#include "PathFinder.hpp"
#include "Commodity.hpp"

using namespace std;

typedef boost::graph_traits<Graph>::edge_descriptor EdgeDescriptor;

// CUDA kernel to calculate bottleneck values for all edges
__global__ void calculateBottleneckKernel(EdgeProperties* edgeProperties, int numEdges, double* bottleneckValues) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numEdges) {
        if (edgeProperties[idx].flow > 0) {
            bottleneckValues[idx] = edgeProperties[idx].capacity / edgeProperties[idx].flow;
        }
        else {
            bottleneckValues[idx] = DBL_MAX; // Max value for edges with no flow
        }
    }
}

// CUDA kernel to normalize flows
__global__ void normalizeFlowsKernel(EdgeProperties* edgeProperties, int numEdges, double bottleneckValue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numEdges) {
        if (edgeProperties[idx].flow > 0) {
            edgeProperties[idx].flow *= bottleneckValue;
        }
    }
}

// CUDA kernel to recalculate weights
__global__ void recalculateWeightsKernel(EdgeProperties* edgeProperties, int numEdges, double alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numEdges) {
        double flowRatio = edgeProperties[idx].flow / edgeProperties[idx].capacity;
        edgeProperties[idx].weight = exp(alpha * flowRatio);
    }
}

// Host function to perform flow distribution algorithm using CUDA
double CUDA_flowDistributionAlgorithm(Graph& g, vector<Commodity>& commodities, double epsilon, double alpha) {
    double solution = 0.0;

    // Extract edge properties and prepare for CUDA
    vector<EdgeProperties> edges;
    for (auto e : boost::make_iterator_range(boost::edges(g))) {
        edges.push_back(g[e]);
    }

    int numEdges = edges.size();
    EdgeProperties* d_edgeProperties;
    double* d_bottleneckValues;

    // Allocate GPU memory
    cudaMalloc(&d_edgeProperties, numEdges * sizeof(EdgeProperties));
    cudaMalloc(&d_bottleneckValues, numEdges * sizeof(double));

    // Copy edge properties to GPU
    cudaMemcpy(d_edgeProperties, edges.data(), numEdges * sizeof(EdgeProperties), cudaMemcpyHostToDevice);

    // Set up CUDA kernel launch configuration
    int blockSize = 256;
    int numBlocks = (numEdges + blockSize - 1) / blockSize;

    double prevMaxRatio = 0.0;
    while (true) {
        // Step 1: Calculate bottleneck value
        calculateBottleneckKernel << <numBlocks, blockSize >> > (d_edgeProperties, numEdges, d_bottleneckValues);

        // Copy bottleneck values back to host and find the minimum
        vector<double> bottleneckValues(numEdges);
        cudaMemcpy(bottleneckValues.data(), d_bottleneckValues, numEdges * sizeof(double), cudaMemcpyDeviceToHost);
        double bottleneckValue = *min_element(bottleneckValues.begin(), bottleneckValues.end());

        // Step 2: Normalize flows
        normalizeFlowsKernel << <numBlocks, blockSize >> > (d_edgeProperties, numEdges, bottleneckValue);

        // Step 3: Recalculate weights
        recalculateWeightsKernel << <numBlocks, blockSize >> > (d_edgeProperties, numEdges, alpha);

        // Step 4: Compute maximum ratio after redistribution
        cudaMemcpy(edges.data(), d_edgeProperties, numEdges * sizeof(EdgeProperties), cudaMemcpyDeviceToHost);
        double maxRatio = 0.0;
        for (const auto& edge : edges) {
            if (edge.flow > 0) {
                maxRatio = max(maxRatio, static_cast<double>(edge.capacity) / edge.flow);
            }
        }

        // Step 5: Check for convergence
        if (abs(maxRatio - prevMaxRatio) < epsilon) {
            solution = maxRatio;
            break;
        }
        prevMaxRatio = maxRatio;
    }

    // Free GPU memory
    cudaFree(d_edgeProperties);
    cudaFree(d_bottleneckValues);

    return solution;
}
