#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <cmath>
#include "NetworkGraph.hpp"
#include "PathFinder.hpp"
#include <cuda_runtime.h>

__device__ __host__ void cudaSendFlow(NetworkGraph& graph, const std::vector<std::string>& path, double amount) {
    for (int i = 0; i < path.size() - 1; ++i) {
        Edge& edge = graph.getEdge(path[i], path[i + 1]);
        edge.flow += amount;
    }
}

__global__ void calculateSuccessRates(double* successRates, double* unitsDelivered, double* demands, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        successRates[idx] = (demands[idx] > 0) ? unitsDelivered[idx] / demands[idx] : 1.0;
    }
}

__global__ void findLowestSuccessRate(double* successRates, int* lowestIndex, int size) {
    __shared__ int localIndex;
    __shared__ double localMin;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadIdx.x == 0) {
        localIndex = 0;
        localMin = successRates[0];
    }

    __syncthreads();

    if (idx < size) {
        if (successRates[idx] < localMin) {
            atomicMin(&localIndex, idx);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *lowestIndex = localIndex;
    }
}

extern "C" void CUDA_equalDistributionAlgorithm(NetworkGraph & graph,
    std::vector<std::pair<std::string, std::string>> commodities,
    std::vector<double> demands) {

    int size = commodities.size();

    // Allocate host memory
    std::vector<double> successRates(size, 0.0);
    std::vector<double> unitsDelivered(size, 0.0);

    // Allocate device memory
    double* d_successRates, * d_unitsDelivered, * d_demands;
    int* d_lowestIndex;

    cudaMalloc(&d_successRates, size * sizeof(double));
    cudaMalloc(&d_unitsDelivered, size * sizeof(double));
    cudaMalloc(&d_demands, size * sizeof(double));
    cudaMalloc(&d_lowestIndex, sizeof(int));

    cudaMemcpy(d_demands, demands.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_unitsDelivered, unitsDelivered.data(), size * sizeof(double), cudaMemcpyHostToDevice);

    bool moreFlowNeeded = true;

    while (moreFlowNeeded) {
        moreFlowNeeded = false;

        // Calculate success rates in parallel
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

        calculateSuccessRates << <blocksPerGrid, threadsPerBlock >> > (d_successRates, d_unitsDelivered, d_demands, size);
        cudaDeviceSynchronize();

        // Find commodity with the lowest success rate in parallel
        findLowestSuccessRate << <blocksPerGrid, threadsPerBlock >> > (d_successRates, d_lowestIndex, size);
        cudaDeviceSynchronize();

        int lowestIndex;
        cudaMemcpy(&lowestIndex, d_lowestIndex, sizeof(int), cudaMemcpyDeviceToHost);

        const std::string& source = commodities[lowestIndex].first;
        const std::string& destination = commodities[lowestIndex].second;
        double remainingDemand = demands[lowestIndex] - unitsDelivered[lowestIndex];

        // Get all paths from source to destination (host-side operation for now)
        std::vector<std::vector<std::string>> allPaths = findAllPaths(graph.getEdges(), source, destination);

        for (const auto& path : allPaths) {
            if (remainingDemand <= 0) break;

            // Calculate bottleneck capacity
            double pathCapacity = std::numeric_limits<double>::max();
            for (size_t j = 0; j < path.size() - 1; ++j) {
                Edge& edge = graph.getEdge(path[j], path[j + 1]);
                pathCapacity = std::min(pathCapacity, static_cast<double>(edge.capacity - edge.flow));
            }

            if (pathCapacity > 0) {
                moreFlowNeeded = true;

                // Allocate flow
                double flowToSend = std::min(remainingDemand, pathCapacity);
                cudaSendFlow(graph, path, flowToSend);

                // Update demand and track delivered flow
                unitsDelivered[lowestIndex] += flowToSend;
                remainingDemand -= flowToSend;
            }
        }

        cudaMemcpy(d_unitsDelivered, unitsDelivered.data(), size * sizeof(double), cudaMemcpyDeviceToHost);
    }

    // Free device memory
    cudaFree(d_successRates);
    cudaFree(d_unitsDelivered);
    cudaFree(d_demands);
    cudaFree(d_lowestIndex);

    std::cout << "CUDA-based proportional flow algorithm completed.\n";
}
