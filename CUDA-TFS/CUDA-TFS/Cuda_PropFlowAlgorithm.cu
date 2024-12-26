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

__global__ void calculate_bottleneck_kernel(double* flow, double* capacity, double* min_ratio, int edge_count) {
    extern __shared__ double shared_min[];  // Shared memory for block-level reduction

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_idx = threadIdx.x;

    // Initialize the local minimum to a high value
    double local_min = std::numeric_limits<double>::max();

    // Each thread processes one edge
    if (idx < edge_count && flow[idx] > 0) {
        double ratio = capacity[idx] / flow[idx];
        local_min = ratio;
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
        atomicMin(min_ratio, shared_min[0]);
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

std::vector<int> cuda_getEdgesWithFlow(const std::vector<double>& flow) {
    int edge_count = flow.size();

    // Device pointers
    double* d_flow;
    int* d_result_indices;
    int* d_result_count;

    // Allocate device memory
    cudaMalloc(&d_flow, edge_count * sizeof(double));
    cudaMalloc(&d_result_indices, edge_count * sizeof(int)); // Maximum possible size
    cudaMalloc(&d_result_count, sizeof(int));

    // Copy data to device
    cudaMemcpy(d_flow, flow.data(), edge_count * sizeof(double), cudaMemcpyHostToDevice);
    int initial_count = 0;
    cudaMemcpy(d_result_count, &initial_count, sizeof(int), cudaMemcpyHostToDevice);

    // Configure the kernel
    int threads_per_block = 256;
    int blocks_per_grid = (edge_count + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    getEdgesWithFlowKernel << <blocks_per_grid, threads_per_block >> > (d_flow, d_result_indices, d_result_count, edge_count);

    // Retrieve the result count
    int result_count;
    cudaMemcpy(&result_count, d_result_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Retrieve the result indices
    std::vector<int> result_indices(result_count);
    cudaMemcpy(result_indices.data(), d_result_indices, result_count * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_flow);
    cudaFree(d_result_indices);
    cudaFree(d_result_count);

    return result_indices;
}


double CUDA_flowDistributionAlgorithm(Graph& g, std::vector<Commodity>& commodities, double epsilon, double alpha) {
    // Step 1: Normalize Flows
    {
        std::cout << "=== Normalizing Flows ===" << std::endl;
        std::vector<double> flow = { 10.0, 20.0, 30.0 }; // Example flow values
        double bottleneck_value = 0.5;                 // Example bottleneck value

        // Normalize flows using CUDA
        cuda_normalize_flows(flow, bottleneck_value);

        // Print the updated flows
        for (double f : flow) {
            std::cout << "Normalized flow: " << f << std::endl;
        }
    }

    // Step 2: Update Commodities Sent
    {
        std::cout << "=== Updating Commodities Sent ===" << std::endl;
        //std::vector<Commodity> commodities = {
        //    {0, 1, 100.0, 50.0},
        //    {1, 2, 200.0, 100.0},
        //    {2, 3, 300.0, 150.0}
        //};
        double bottleneck_value = 0.5; // Example bottleneck value

        // Update commodities sent values using CUDA
        cuda_updateCommoditiesSent(commodities, bottleneck_value);

        // Print the updated commodities
        for (const auto& commodity : commodities) {
            std::cout << "Commodity sent: " << commodity.sent << std::endl;
        }
    }

    // Step 3: Recalculate Weights
    {
        std::cout << "=== Recalculating Weights ===" << std::endl;
        std::vector<double> flow = { 10.0, 20.0, 30.0 };      // Example flow values
        std::vector<double> capacity = { 50.0, 40.0, 60.0 };  // Example capacity values
        std::vector<double> weights = { 1.0, 1.0, 1.0 };      // Initial weights
        double alpha = 0.1;                                 // Example alpha value

        // Recalculate weights using CUDA
        cuda_recalculate_weights(flow, capacity, weights, alpha);

        // Print the updated weights
        for (double weight : weights) {
            std::cout << "Weight: " << weight << std::endl;
        }
    }

    // Step 4: Check Flow Exceeding Capacity
    {
        std::cout << "=== Checking Flow Exceeding Capacity ===" << std::endl;
        std::vector<double> flow = { 10.0, 20.0, 30.0 };      // Example flow values
        std::vector<double> capacity = { 50.0, 15.0, 60.0 }; // Example capacity values

        // Check if any flow exceeds capacity
        int exceedsCapacity = cuda_isFlowExceedingCapacity(flow, capacity);

        if (exceedsCapacity) {
            std::cout << "At least one edge's flow exceeds its capacity!" << std::endl;
        }
        else {
            std::cout << "All edges are within capacity limits." << std::endl;
        }
    }

    // Step 5: Get Edges with Flow > 0
    {
        std::cout << "=== Getting Edges with Flow > 0 ===" << std::endl;
        // Example graph and flow
        std::vector<double> flow; // Populate with flow values for edges

        // Get edges with flow > 0 using CUDA
        std::vector<int> edge_indices_with_flow = cuda_getEdgesWithFlow(flow);

        // Map indices back to edge descriptors
        std::vector<boost::graph_traits<Graph>::edge_descriptor> edges_with_flow;
        auto edges = boost::edges(g);
        for (int idx : edge_indices_with_flow) {
            edges_with_flow.push_back(*(edges.first + idx)); // Retrieve edge descriptor
        }

        // Print edges with flow
        for (auto e : edges_with_flow) {
            std::cout << "Edge with flow: " << boost::source(e, g) << " -> " << boost::target(e, g) << std::endl;
        }
    }

    return 0;
}
