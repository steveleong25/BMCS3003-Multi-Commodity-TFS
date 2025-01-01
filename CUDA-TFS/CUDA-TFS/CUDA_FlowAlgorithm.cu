#include <cuda_runtime.h>
#include "NetworkGraph.hpp"
#include "Commodity.hpp"
#include "PathFinder.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;
__device__ const double MAX_WEIGHT = 1e9;
__device__ const double DECAY = 0.9;

__global__ void calculate_path_weights_kernel(const int* path, const double* edge_weights, int path_length, double* result) {
    // Shared memory to hold partial sums
    extern __shared__ double shared_weights[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    // Ensure we don’t go out of bounds
    if (index < path_length - 1) {
        // Compute the weight for this segment of the path
        int from = path[index];
        int to = path[index + 1];
        shared_weights[tid] = edge_weights[from * gridDim.x + to];
    }
    else {
        shared_weights[tid] = 0.0; // Zero padding for unused threads
    }

    __syncthreads();

    // Reduce the weights in shared memory to get the total weight
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_weights[tid] += shared_weights[tid + stride];
        }
        __syncthreads();
    }

    // The first thread in the block writes the result
    if (tid == 0) {
        atomicAdd(result, shared_weights[0]);
    }
}

double cuda_calculate_path_weight(const Graph& g, const std::vector<int>& path) {
    int path_length = path.size();

    // Allocate host memory
    int* h_path = new int[path_length];
    int num_vertices = boost::num_vertices(g);
    double* h_edge_weights = new double[num_vertices * num_vertices];
    double h_result = 0.0;

    // Fill host memory with graph data
    for (int i = 0; i < path_length; ++i) {
        h_path[i] = path[i];
    }
    for (const auto& edge : boost::make_iterator_range(boost::edges(g))) {
        auto source = boost::source(edge, g);
        auto target = boost::target(edge, g);
        h_edge_weights[source * num_vertices + target] = g[edge].weight;
    }

    // Allocate device memory
    int* d_path;
    double* d_edge_weights;
    double* d_result;
    cudaMalloc(&d_path, path_length * sizeof(int));
    cudaMalloc(&d_edge_weights, num_vertices * num_vertices * sizeof(double));
    cudaMalloc(&d_result, sizeof(double));

    // Copy data from host to device
    cudaMemcpy(d_path, h_path, path_length * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_weights, h_edge_weights, num_vertices * num_vertices * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(double), cudaMemcpyHostToDevice);

    // Launch the kernel
    int threads_per_block = 256;
    int blocks_per_grid = (path_length + threads_per_block - 1) / threads_per_block;
    calculate_path_weights_kernel << <blocks_per_grid, threads_per_block, threads_per_block * sizeof(double) >> > (d_path, d_edge_weights, path_length, d_result);

    // Copy the result back to host
    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_path);
    cudaFree(d_edge_weights);
    cudaFree(d_result);

    // Free host memory
    delete[] h_path;
    delete[] h_edge_weights;

    return h_result;
}



vector<boost::graph_traits<Graph>::edge_descriptor> cuda_get_edges_with_flow(Graph& g) {
    vector<boost::graph_traits<Graph>::edge_descriptor> edges_with_flow;

    for (auto e : boost::make_iterator_range(boost::edges(g))) {
        if (g[e].flow > 0) {
            edges_with_flow.push_back(e);
        }
    }

    return edges_with_flow;
}

void CUDA_flowDistributionAlgorithm(Graph& g, vector<Commodity>& commodities, int num_of_iter) {
    for (int i = 0; i < num_of_iter; i++) {
        if (i == num_of_iter - 1)
            cout << "Iteration " << i << endl;
        for (int j = 0; j < commodities.size(); j++) {
            // Retrieve all shortest paths for all source-destination pair
            std::vector<std::vector<std::vector<int>>> all_shortest_paths = find_all_shortest_paths(g);

            const std::vector<int>& path = all_shortest_paths[commodities[j].source][commodities[j].destination];

            if (cuda_calculate_path_weight(g, path) >= DBL_MAX) {
                break;
            }

            if (path.empty()) {
                std::cerr << "No path exists between source " << commodities[j].source
                    << " and destination " << commodities[j].destination << std::endl;
                continue;
            }

            // calculate delta flow
            double delta_flow = commodities[j].demand / num_of_iter;

            double bottleneck_capacity = std::numeric_limits<double>::infinity();

            // Use the shortest path to distribute flows
            for (size_t p = 1; p < path.size(); ++p) {
                auto e = boost::edge(path[p - 1], path[p], g).first;
                double local_cap = g[e].capacity;
                double local_flow = g[e].flow;

                double available_capacity = local_cap - local_flow;

                bottleneck_capacity = std::min(bottleneck_capacity, available_capacity);
            }

            double total_flow_assigned = 0.0;

            // Distribute flow along the path, limited by the bottleneck capacity
            for (size_t p = 1; p < path.size(); ++p) {
                auto e = boost::edge(path[p - 1], path[p], g).first;

                double flow_to_assign = std::min(delta_flow, bottleneck_capacity);

                // update the flow for both the forward and reverse directions
                g[e].flow += flow_to_assign;

                // update the reverse edge
                auto reverse_edge = boost::edge(path[p], path[p - 1], g).first;
                g[reverse_edge].flow -= flow_to_assign;

                // calculate the new weight for the forward edge
                double local_cap = g[e].capacity;
                double local_flow = g[e].flow;
                double new_weight = MAX_WEIGHT * exp(-DECAY * abs(local_cap - local_flow));
                g[e].weight = new_weight;

                // calculate the new weight for the reverse edge
                local_cap = g[reverse_edge].capacity;
                local_flow = g[reverse_edge].flow;
                new_weight = MAX_WEIGHT * exp(-DECAY * abs(local_cap - local_flow));
                g[reverse_edge].weight = new_weight;

                total_flow_assigned = flow_to_assign;
            }

            bool path_exists = false;
            for (auto& pair : commodities[j].used_paths_with_flows) {
                if (pair.first == path) { // Check if the path already exists
                    pair.second += total_flow_assigned; // Update the flow for the existing path
                    path_exists = true;
                    break;
                }
            }

            if (!path_exists) {
                commodities[j].used_paths_with_flows.emplace_back(path, total_flow_assigned); // Add a new path and flow
            }

            commodities[j].sent += total_flow_assigned; // Update the total sent flow for the commodity
        }
    }
}