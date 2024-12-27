//// Cuda_PropFlowAlgorithm.cu


#include "NetworkGraph.hpp"
#include "PathFinder.hpp"
#include "Commodity.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <device_launch_parameters.h>

using namespace std;


///////////////////////////// CUDA IMPLEMENTATION //////////////////////////////////////
__device__ double atomicMinDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmin(__longlong_as_double(assumed), val)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void calculate_bottleneck_kernel(double* flow, double* capacity, double* min_ratio, int edge_count) {
    extern __shared__ double shared_min[];  // Shared memory for block-level reduction

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_idx = threadIdx.x;

    // Initialize the local minimum to a high value
    double local_min = DBL_MAX;

    // Each thread processes one edge
    if (idx < edge_count && flow[idx] > 0) {
        double ratio = capacity[idx] / flow[idx];
        local_min = fmin(local_min, ratio);
    }

    // Store the local minimum in shared memory
    shared_min[thread_idx] = local_min;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread_idx < stride) {
            shared_min[thread_idx] = fmin(shared_min[thread_idx], shared_min[thread_idx + stride]);
        }
        __syncthreads();
    }

    // Store the block's minimum value in the global output
    if (thread_idx == 0) {
        atomicMinDouble(min_ratio, shared_min[0]);
    }
}

__global__ void normalize_flows_kernel(double* flow, double bottleneck_value, int edge_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    if (idx < edge_count) {
        flow[idx] *= bottleneck_value; // Multiply flow by the bottleneck value
    }
}

__global__ void updateCommoditiesSentKernel(Commodity* commodities, double bottleneck_value, int num_commodities) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    if (idx < num_commodities) {
        commodities[idx].sent *= bottleneck_value; // Update sent field
    }
}

__global__ void recalculate_weights_kernel(double* flow, double* capacity, double* weights, double alpha, int edge_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    if (idx < edge_count) {
        double flow_ratio = flow[idx] / capacity[idx];
        weights[idx] = exp(alpha * flow_ratio); // Exponential weight calculation
    }
}

__global__ void isFlowExceedingCapacityKernel(double* flow, double* capacity, int* result, int edge_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index

    if (idx < edge_count) {
        if (flow[idx] > capacity[idx]) {
            // Use atomic operation to set the result flag to true
            atomicExch(result, 1);
        }
    }
}

__global__ void getEdgesWithFlowKernel(double* flow, int* result_indices, int* result_count, int edge_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index

    if (idx < edge_count) {
        if (flow[idx] > 0) {
            // Atomically append the index to the result array
            int position = atomicAdd(result_count, 1);
            result_indices[position] = idx;
        }
    }
}


double cuda_calculate_bottleneck(const std::vector<double>& flow, const std::vector<double>& capacity, int edge_count) {
    double* d_flow, * d_capacity, * d_min_ratio;

    // Allocate device memory
    cudaMalloc(&d_flow, edge_count * sizeof(double));
    cudaMalloc(&d_capacity, edge_count * sizeof(double));
    cudaMalloc(&d_min_ratio, sizeof(double));

    // Copy data to device
    cudaMemcpy(d_flow, flow.data(), edge_count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_capacity, capacity.data(), edge_count * sizeof(double), cudaMemcpyHostToDevice);
    double initial_min = std::numeric_limits<double>::max();
    cudaMemcpy(d_min_ratio, &initial_min, sizeof(double), cudaMemcpyHostToDevice);

    // Kernel configuration
    int threads_per_block = 256;
    int blocks_per_grid = (edge_count + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    calculate_bottleneck_kernel << <blocks_per_grid, threads_per_block, threads_per_block * sizeof(double) >> > (
        d_flow, d_capacity, d_min_ratio, edge_count);

    // Retrieve the result
    double min_ratio;
    cudaMemcpy(&min_ratio, d_min_ratio, sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_flow);
    cudaFree(d_capacity);
    cudaFree(d_min_ratio);

    return min_ratio;
}

void cuda_normalize_flows(std::vector<double>& flow, double bottleneck_value) {
    int edge_count = flow.size();
    double* d_flow;

    // Allocate device memory
    cudaMalloc(&d_flow, edge_count * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_flow, flow.data(), edge_count * sizeof(double), cudaMemcpyHostToDevice);

    // Configure the kernel
    int threads_per_block = 256;
    int blocks_per_grid = (edge_count + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    normalize_flows_kernel << <blocks_per_grid, threads_per_block >> > (d_flow, bottleneck_value, edge_count);

    // Copy updated data back to host
    cudaMemcpy(flow.data(), d_flow, edge_count * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_flow);
}

void cuda_updateCommoditiesSent(std::vector<Commodity>& commodities, double bottleneck_value) {
    int num_commodities = commodities.size();
    Commodity* d_commodities;

    // Allocate device memory
    cudaMalloc(&d_commodities, num_commodities * sizeof(Commodity));

    // Copy data to device
    cudaMemcpy(d_commodities, commodities.data(), num_commodities * sizeof(Commodity), cudaMemcpyHostToDevice);

    // Configure the kernel
    int threads_per_block = 256;
    int blocks_per_grid = (num_commodities + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    updateCommoditiesSentKernel << <blocks_per_grid, threads_per_block >> > (d_commodities, bottleneck_value, num_commodities);

    // Copy updated data back to host
    cudaMemcpy(commodities.data(), d_commodities, num_commodities * sizeof(Commodity), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_commodities);
}

void cuda_recalculate_weights(std::vector<double>& flow, std::vector<double>& capacity, std::vector<double>& weights, double alpha) {
    int edge_count = flow.size();

    // Device pointers
    double* d_flow, * d_capacity, * d_weights;

    // Allocate device memory
    cudaMalloc(&d_flow, edge_count * sizeof(double));
    cudaMalloc(&d_capacity, edge_count * sizeof(double));
    cudaMalloc(&d_weights, edge_count * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_flow, flow.data(), edge_count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_capacity, capacity.data(), edge_count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), edge_count * sizeof(double), cudaMemcpyHostToDevice);

    // Configure the kernel
    int threads_per_block = 256;
    int blocks_per_grid = (edge_count + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    recalculate_weights_kernel << <blocks_per_grid, threads_per_block >> > (d_flow, d_capacity, d_weights, alpha, edge_count);

    // Copy updated weights back to host
    cudaMemcpy(weights.data(), d_weights, edge_count * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_flow);
    cudaFree(d_capacity);
    cudaFree(d_weights);
}

bool cuda_isFlowExceedingCapacity(const std::vector<double>& flow, const std::vector<double>& capacity) {
    int edge_count = flow.size();

    // Device pointers
    double* d_flow, * d_capacity;
    int* d_result;

    // Allocate device memory
    cudaMalloc(&d_flow, edge_count * sizeof(double));
    cudaMalloc(&d_capacity, edge_count * sizeof(double));
    cudaMalloc(&d_result, sizeof(bool));

    // Copy data to device
    cudaMemcpy(d_flow, flow.data(), edge_count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_capacity, capacity.data(), edge_count * sizeof(double), cudaMemcpyHostToDevice);

    // Initialize result flag on the device
    int initial_result = 0;
    cudaMemcpy(d_result, &initial_result, sizeof(bool), cudaMemcpyHostToDevice);

    // Configure the kernel
    int threads_per_block = 256;
    int blocks_per_grid = (edge_count + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    isFlowExceedingCapacityKernel << <blocks_per_grid, threads_per_block >> > (d_flow, d_capacity, d_result, edge_count);

    // Copy the result flag back to the host
    int result;
    cudaMemcpy(&result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_flow);
    cudaFree(d_capacity);
    cudaFree(d_result);

    return result;
}

void extractFlowCapacityWeights(Graph& g,
    const std::vector<boost::graph_traits<Graph>::edge_descriptor>& edges_with_flow,
    std::vector<double>& flow,
    std::vector<double>& capacity,
    std::vector<double>& weights) {
    flow.clear();
    capacity.clear();
    weights.clear();
    for (auto& e : edges_with_flow) {
        flow.push_back(g[e].flow);
        capacity.push_back(g[e].capacity);
        weights.push_back(g[e].weight);
    }
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

double CUDA_flowDistributionAlgorithm(Graph& g, std::vector<Commodity>& commodities, double epsilon, double alpha) {
    double solution = 0.0;

    // Step 1: Find shortest path for each commodity and distribute initial flow
    for (auto& commodity : commodities) {
        std::vector<int> path = find_shortest_path(g, commodity.source, commodity.destination);

        for (size_t i = 1; i < path.size(); ++i) {
            auto e = boost::edge(path[i - 1], path[i], g).first;
            g[e].flow += commodity.demand;
            commodity.sent = g[e].flow;
        }
    }

    vector<boost::graph_traits<Graph>::edge_descriptor> edges_with_flow = cuda_get_edges_with_flow(g);

    std::vector<double> flow, capacity, weights;
    extractFlowCapacityWeights(g, edges_with_flow, flow, capacity, weights);

    double prev_max_ratio = 0.0;
    while (true) {
        // Step 2: Calculate the bottleneck value using CUDA
        double bottleneck_value = cuda_calculate_bottleneck(flow, capacity, flow.size());

        // Step 3: Normalize flows if they exceed capacity
        if (cuda_isFlowExceedingCapacity(flow, capacity)) {
            cuda_normalize_flows(flow, bottleneck_value);
            cuda_updateCommoditiesSent(commodities, bottleneck_value);

            // Update the graph's flow values
            for (size_t i = 0; i < edges_with_flow.size(); ++i) {
                g[edges_with_flow[i]].flow = flow[i];
            }
        }

        // Step 4: Print edges with flow details (Optional for debugging)
        for (size_t i = 0; i < edges_with_flow.size(); ++i) {
            auto source_node = boost::source(edges_with_flow[i], g);
            auto target_node = boost::target(edges_with_flow[i], g);

            std::cout << source_node << " -> " << target_node
                << " [Flow: " << static_cast<int>(flow[i]) // Convert flow to integer
                << ", Capacity: " << capacity[i] << "]\n";
        }

        // Step 5: Recalculate weights using CUDA
        cuda_recalculate_weights(flow, capacity, weights, alpha);

        // Update the graph's weight values
        for (size_t i = 0; i < edges_with_flow.size(); ++i) {
            g[edges_with_flow[i]].weight = weights[i];
        }

        // Step 6: Compute the maximum ratio after redistribution
        double max_ratio = 0.0;
        double highest_flow = 0.0, min_capacity = 0.0;

        for (size_t i = 0; i < edges_with_flow.size(); ++i) {
            double total_flow_on_edge = static_cast<int>(flow[i]);
            double edge_capacity = capacity[i];

            if (total_flow_on_edge > highest_flow) {
                highest_flow = total_flow_on_edge;
                min_capacity = edge_capacity;
                max_ratio = edge_capacity / total_flow_on_edge;
            }
            else if (total_flow_on_edge == highest_flow) {
                if (edge_capacity < min_capacity) {
                    min_capacity = edge_capacity;
                    max_ratio = edge_capacity / total_flow_on_edge;
                }
            }
        }

        // Step 7: Check for convergence
        if (std::abs(max_ratio - prev_max_ratio) < epsilon) {
            solution = max_ratio;
            break;
        }
        prev_max_ratio = max_ratio;
    }

    return solution;
}
