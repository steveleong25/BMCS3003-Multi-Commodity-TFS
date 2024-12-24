#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include "NetworkGraph.hpp"
#include "PathFinder.hpp"
#include "Commodity.hpp"
#include <device_launch_parameters.h>


using namespace std;

__global__ void calculate_bottleneck_kernel(const int* edge_usage, const int* capacities, int num_edges, int* block_results) {
    extern __shared__ int shared_data[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    // Initialize shared memory
    shared_data[tid] = 0;
    if (idx < num_edges) {
        shared_data[tid] = edge_usage[idx] * capacities[idx];
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }

    // Write the result of this block to the global memory
    if (tid == 0) {
        block_results[blockIdx.x] = shared_data[0];
    }
}

void cudaCalculate_bottleneck(const int* edge_usage, const int* capacities, int num_edges, int& bottleneck_value) {
    int blockSize = 256;
    int gridSize = (num_edges + blockSize - 1) / blockSize;
    int* d_block_results;
    int* h_block_results = new int[gridSize];

    cudaMalloc(&d_block_results, gridSize * sizeof(int));

    calculate_bottleneck_kernel << <gridSize, blockSize, blockSize * sizeof(int) >> > (edge_usage, capacities, num_edges, d_block_results);

    cudaMemcpy(h_block_results, d_block_results, gridSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_block_results);

    // Final reduction on the host
    bottleneck_value = 0;
    for (int i = 0; i < gridSize; ++i) {
        bottleneck_value = max(bottleneck_value, h_block_results[i]);
    }

    delete[] h_block_results;
}

__global__ void normalize_flows_kernel(int* flows, const int* capacities, int num_edges, double bottleneck_value) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_edges) {
        if (flows[idx] > capacities[idx]) {
            flows[idx] = flows[idx] / bottleneck_value;
        }
    }
}

__global__ void recalculate_weights_kernel(double* weights, const int* flows, double alpha, int num_edges) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_edges) {
        weights[idx] = exp(alpha * flows[idx]);
    }
}

double cudaFlowDistributionAlgorithm(Graph & g, vector<Commodity>&commodities, double epsilon, double alpha) {
    int num_edges = boost::num_edges(g);
    int num_vertices = boost::num_vertices(g);

    vector<vector<int>> assigned_paths;
    double total_demand = 0.0;

    // Initialize arrays for GPU
    int* d_edge_usage;
    int* d_flows;
    int* d_capacities;
    double* d_weights;

    cudaMalloc(&d_edge_usage, num_edges * sizeof(int));
    cudaMalloc(&d_flows, num_edges * sizeof(int));
    cudaMalloc(&d_capacities, num_edges * sizeof(int));
    cudaMalloc(&d_weights, num_edges * sizeof(double));

    vector<int> capacities(num_edges, 0);
    vector<int> flows(num_edges, 0);
    vector<double> weights(num_edges, 0.0);

    // Copy initial capacities and flows from graph
    boost::graph_traits<Graph>::edge_iterator ei, ei_end;
    int idx = 0;
    for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei, ++idx) {
        capacities[idx] = g[*ei].capacity;
        flows[idx] = g[*ei].flow;
        weights[idx] = 1.0;  // Initialize weights to 1.0
    }


    cudaMemcpy(d_capacities, capacities.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flows, flows.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), num_edges * sizeof(double), cudaMemcpyHostToDevice);

    double prev_max_ratio = 0.0;
    while (true) {
        int bottleneck_value = 0;
        cudaCalculate_bottleneck(d_edge_usage, capacities.data(), num_edges, bottleneck_value);

        // Normalize flows
        int blockSize = 256;
        int gridSize = (num_edges + blockSize - 1) / blockSize;
        normalize_flows_kernel << <gridSize, blockSize >> > (d_flows, d_capacities, num_edges, bottleneck_value);

        // Recalculate weights
        recalculate_weights_kernel << <gridSize, blockSize >> > (d_weights, d_flows, alpha, num_edges);

        // Check for convergence
        double max_ratio = static_cast<double>(bottleneck_value) / total_demand;
        if (fabs(max_ratio - prev_max_ratio) < epsilon) {
            cudaMemcpy(flows.data(), d_flows, num_edges * sizeof(int), cudaMemcpyDeviceToHost);
            break;
        }
        prev_max_ratio = max_ratio;

        // Update flows for next iteration
        for (auto& commodity : commodities) {
            vector<int> path = find_shortest_path(g, commodity.source, commodity.destination);
            assigned_paths.push_back(path);

            for (int i = 1; i < path.size(); ++i) {

                auto e = boost::edge(path[i - 1], path[i], g).first;

                std::cout << "Path size: " << path.size() << ", Current i: " << i << std::endl;
                std::cout << "Flow index: " << idx << std::endl;
                //if (idx < 0 || idx >= flows.size()) {
                //    std::cerr << "Error: Index out of range. idx = " << idx << ", size = " << flows.size() << std::endl;
                //    continue;  // Skip this iteration
                //}
                flows[idx] += commodity.demand;
            }
        }
        cudaMemcpy(d_flows, flows.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Clean up
    cudaFree(d_edge_usage);
    cudaFree(d_flows);
    cudaFree(d_capacities);
    cudaFree(d_weights);

    return prev_max_ratio;
}
