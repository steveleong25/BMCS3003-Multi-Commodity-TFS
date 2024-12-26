#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "NetworkGraph.hpp"
#include "PathFinder.hpp"
#include "Commodity.hpp"

using namespace std;

typedef boost::graph_traits<Graph>::edge_descriptor EdgeDescriptor;

// CUDA Kernel for bottleneck calculation
__global__ void calculate_bottleneck_kernel(double* capacities, double* flows, int num_edges, double* bottleneck_array) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < num_edges && flows[idx] > 0) {
        bottleneck_array[idx] = capacities[idx] / flows[idx];
    }
    else if (idx < num_edges) {
        bottleneck_array[idx] = DBL_MAX;
    }
}

// CUDA Kernel for finding minimum bottleneck
__global__ void reduce_min_kernel(double* bottleneck_array, int num_edges, double* result) {
    extern __shared__ double shared_data[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    if (idx < num_edges) {
        shared_data[tid] = bottleneck_array[idx];
    }
    else {
        shared_data[tid] = DBL_MAX;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] = min(shared_data[tid], shared_data[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = shared_data[0];
    }
}

// CUDA Kernel for normalizing flows
__global__ void normalize_flows_kernel(double* flows, int num_edges, double bottleneck, double* capacities) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < num_edges && flows[idx] > 0) {
        flows[idx] = min(flows[idx] * bottleneck, capacities[idx]);
    }
}

// CUDA Kernel for recalculating weights
__global__ void recalculate_weights_kernel(double* flows, double* capacities, double* weights, int num_edges, double alpha) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < num_edges) {
        double flow_ratio = flows[idx] / capacities[idx];
        weights[idx] = exp(alpha * flow_ratio);
    }
}

double CUDA_flowDistributionAlgorithm(Graph& g, vector<Commodity>& commodities, double epsilon, double alpha) {
    int num_edges = boost::num_edges(g);

    // Allocate and copy graph data to device
    double* d_capacities;
    double* d_flows;
    double* d_weights;
    double* d_bottleneck_array;
    double* d_bottleneck_results;

    cudaMalloc(&d_capacities, num_edges * sizeof(double));
    cudaMalloc(&d_flows, num_edges * sizeof(double));
    cudaMalloc(&d_weights, num_edges * sizeof(double));
    cudaMalloc(&d_bottleneck_array, num_edges * sizeof(double));
    cudaMalloc(&d_bottleneck_results, sizeof(double) * ((num_edges + 255) / 256));

    vector<double> capacities(num_edges), flows(num_edges), weights(num_edges);
    int i = 0;
    for (auto e : boost::make_iterator_range(boost::edges(g))) {
        capacities[i] = g[e].capacity;
        flows[i] = g[e].flow;
        weights[i] = g[e].weight;
        i++;
    }

    cudaMemcpy(d_capacities, capacities.data(), num_edges * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flows, flows.data(), num_edges * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), num_edges * sizeof(double), cudaMemcpyHostToDevice);

    // Initialize flows for commodities
    for (auto& commodity : commodities) {
        std::vector<int> path = find_shortest_path(g, commodity.source, commodity.destination);
        for (size_t i = 1; i < path.size(); ++i) {
            auto e = boost::edge(path[i - 1], path[i], g).first;
            g[e].flow += commodity.demand;
            commodity.sent = min(commodity.demand, commodity.sent + g[e].flow); // Restrict sent to demand
        }
    }

    cudaMemcpy(d_flows, flows.data(), num_edges * sizeof(double), cudaMemcpyHostToDevice); // Ensure flows are copied to device

    double solution = 0.0;
    double prev_max_ratio = 0.0;

    while (true) {
        // Step 1: Calculate the bottleneck value
        int threads_per_block = 256;
        int blocks = (num_edges + threads_per_block - 1) / threads_per_block;

        calculate_bottleneck_kernel << <blocks, threads_per_block >> > (d_capacities, d_flows, num_edges, d_bottleneck_array);
        cudaDeviceSynchronize();

        reduce_min_kernel << <blocks, threads_per_block, threads_per_block * sizeof(double) >> > (d_bottleneck_array, num_edges, d_bottleneck_results);
        cudaDeviceSynchronize();

        double bottleneck = DBL_MAX;
        vector<double> bottleneck_results(blocks);
        cudaMemcpy(bottleneck_results.data(), d_bottleneck_results, blocks * sizeof(double), cudaMemcpyDeviceToHost);

        for (double val : bottleneck_results) {
            bottleneck = min(bottleneck, val);
        }

        // Debugging: Print edge states
        cudaMemcpy(flows.data(), d_flows, num_edges * sizeof(double), cudaMemcpyDeviceToHost);
        cout << "Edge states after bottleneck calculation:" << endl;
        for (int i = 0; i < num_edges; ++i) {
            cout << "Edge " << i << ": Capacity = " << capacities[i] << ", Flow = " << flows[i] << endl;
        }

        // Step 2: Normalize flows using the bottleneck value
        normalize_flows_kernel << <blocks, threads_per_block >> > (d_flows, num_edges, bottleneck, d_capacities);
        cudaDeviceSynchronize();

        // Step 3: Recalculate weights
        recalculate_weights_kernel << <blocks, threads_per_block >> > (d_flows, d_capacities, d_weights, num_edges, alpha);
        cudaDeviceSynchronize();

        // Step 4: Compute the maximum ratio after redistribution
        double max_ratio = 0.0;
        cudaMemcpy(flows.data(), d_flows, num_edges * sizeof(double), cudaMemcpyDeviceToHost);
        for (int i = 0; i < num_edges; ++i) {
            if (flows[i] > 0) {
                double ratio = capacities[i] / flows[i];
                max_ratio = max(max_ratio, ratio);
            }
        }
        cout << "Maximum Ratio: " << max_ratio << endl;

        // Iterate through all edges in the graph
        cout << "Edge states after normalization and weight recalculation:" << endl;
        for (int i = 0; i < num_edges; ++i) {
            cout << "Edge " << i << ": Capacity = " << capacities[i] << ", Flow = " << flows[i] << endl;
        }

        // Step 5: Check for convergence
        if (abs(max_ratio - prev_max_ratio) < epsilon) {
            solution = max_ratio;
            break;
        }
        prev_max_ratio = max_ratio;
    }

    // Update sent for each commodity based on final flows
    for (auto& commodity : commodities) {
        commodity.sent = 0.0;
        std::vector<int> path = find_shortest_path(g, commodity.source, commodity.destination);
        for (size_t i = 1; i < path.size(); ++i) {
            auto e = boost::edge(path[i - 1], path[i], g).first;
            commodity.sent = min(commodity.demand, commodity.sent + g[e].flow); // Restrict sent to demand
        }
    }

    // Free device memory
    cudaFree(d_capacities);
    cudaFree(d_flows);
    cudaFree(d_weights);
    cudaFree(d_bottleneck_array);
    cudaFree(d_bottleneck_results);

    return solution;
}
