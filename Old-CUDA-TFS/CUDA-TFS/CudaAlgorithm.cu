#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include "NetworkGraph.hpp"
#include "PathFinder.hpp"
#include <algorithm>
#include <cmath>

using namespace std;

__host__ void cudaSendFlow(NetworkGraph& graph, const vector<string>& path, double amount) {
    for (int i = 0; i < path.size() - 1; ++i) {
        Edge& edge = graph.getEdge(path[i], path[i + 1]);
        edge.flow += amount;
    }
}


__device__ double atomicSubDouble(double* address, double value) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(__longlong_as_double(assumed) - value));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void calculateFlow(double* d_unitsDelivered, double* d_demands, double* d_successRates,
    double* d_totalUnitsDelivered, double* d_totalDemand, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        atomicAdd(d_totalUnitsDelivered, d_unitsDelivered[idx]);
        atomicAdd(d_totalDemand, d_demands[idx]);
    }
}

__global__ void updateSuccessRates(double* d_unitsDelivered, double* d_demands, double* d_successRates, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_successRates[idx] = (d_demands[idx] > 0) ? d_unitsDelivered[idx] / d_demands[idx] : 1.0;
    }
}

__global__ void redistributeExcessFlow(double* d_unitsDelivered, double* d_demands, double* d_successRates,
    double equalSuccessRate, double* d_unusedFlow, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && d_successRates[idx] > equalSuccessRate) {
        double excessFlow = floor((d_successRates[idx] - equalSuccessRate) * d_demands[idx]);
        d_unitsDelivered[idx] -= excessFlow;
        atomicAdd(&d_unusedFlow[idx], excessFlow);
    }
}

__global__ void redistributeDeficitFlow(double* d_unitsDelivered, double* d_demands, double* d_successRates,
    double equalSuccessRate, double* d_unusedFlow, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && d_successRates[idx] < equalSuccessRate) {
        double neededFlow = ceil((equalSuccessRate - d_successRates[idx]) * d_demands[idx]);
        double assignedFlow = min(neededFlow, *d_unusedFlow);
        d_unitsDelivered[idx] += assignedFlow;
        atomicSubDouble(d_unusedFlow, assignedFlow);
    }
}

__global__ void finalizeRedistribution(double* d_unitsDelivered, double* d_demands, double* d_successRates,
    double* d_unusedFlow, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && *d_unusedFlow > 0) {
        double totalDeficit = 0.0;
        __shared__ double sharedDeficit[256];

        // Calculate each commodity's deficit proportionally
        if (idx < size && d_successRates[idx] < 1.0) {
            sharedDeficit[threadIdx.x] = (1.0 - d_successRates[idx]) * d_demands[idx];
        }
        else {
            sharedDeficit[threadIdx.x] = 0.0;
        }

        __syncthreads();

        // Sum all deficits
        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            if (threadIdx.x % (2 * stride) == 0) {
                sharedDeficit[threadIdx.x] += sharedDeficit[threadIdx.x + stride];
            }
            __syncthreads();
        }

        totalDeficit = sharedDeficit[0];

        // Distribute unused flow proportionally based on deficits
        if (totalDeficit > 0.0 && d_successRates[idx] < 1.0) {
            double proportionalFlow = (*d_unusedFlow * ((1.0 - d_successRates[idx]) * d_demands[idx])) / totalDeficit;
            d_unitsDelivered[idx] += proportionalFlow;
            atomicSubDouble(d_unusedFlow, proportionalFlow);
        }
    }
}

__global__ void forceRedistributeRemainingFlow(double* d_unitsDelivered, double* d_demands, double* d_unusedFlow, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && *d_unusedFlow > 0) {
        double flowToAdd = *d_unusedFlow / size;
        d_unitsDelivered[idx] += flowToAdd;
        atomicSubDouble(d_unusedFlow, flowToAdd);
    }
}

void CudaRedistributeFlow(NetworkGraph& graph, vector<pair<string, string>>& commodities,
    vector<double>& demands, vector<double>& unitsDelivered,
    vector<double>& successRates) {
    int size = commodities.size();

    // Allocate device memory
    double* d_unitsDelivered, * d_demands, * d_successRates, * d_unusedFlow;
    double* d_totalUnitsDelivered, * d_totalDemand;

    cudaMalloc((void**)&d_unitsDelivered, size * sizeof(double));
    cudaMalloc((void**)&d_demands, size * sizeof(double));
    cudaMalloc((void**)&d_successRates, size * sizeof(double));
    cudaMalloc((void**)&d_unusedFlow, sizeof(double));
    cudaMalloc((void**)&d_totalUnitsDelivered, sizeof(double));
    cudaMalloc((void**)&d_totalDemand, sizeof(double));

    // Copy data to device
    cudaMemcpy(d_unitsDelivered, unitsDelivered.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_demands, demands.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_successRates, successRates.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_unusedFlow, 0, sizeof(double));

    double totalUnitsDelivered = 0, totalDemand = 0;
    cudaMemcpy(d_totalUnitsDelivered, &totalUnitsDelivered, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_totalDemand, &totalDemand, sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Calculate total units delivered and demand
    calculateFlow << <blocksPerGrid, threadsPerBlock >> > (d_unitsDelivered, d_demands, d_successRates,
        d_totalUnitsDelivered, d_totalDemand, size);

    cudaMemcpy(&totalUnitsDelivered, d_totalUnitsDelivered, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&totalDemand, d_totalDemand, sizeof(double), cudaMemcpyDeviceToHost);

    double equalSuccessRate = totalUnitsDelivered / totalDemand;

    bool redistributionNeeded = true;
    int maxIterations = 1000;
    int iteration = 0;

    while (redistributionNeeded&& iteration < maxIterations) {
        iteration++;

        // Update success rates
        updateSuccessRates << <blocksPerGrid, threadsPerBlock >> > (d_unitsDelivered, d_demands, d_successRates, size);

        // Redistribute excess flow
        redistributeExcessFlow << <blocksPerGrid, threadsPerBlock >> > (d_unitsDelivered, d_demands, d_successRates, equalSuccessRate, d_unusedFlow, size);

        // Redistribute deficit flow
        redistributeDeficitFlow << <blocksPerGrid, threadsPerBlock >> > (d_unitsDelivered, d_demands, d_successRates, equalSuccessRate, d_unusedFlow, size);

        // Check unused flow
        double totalUnusedFlow;
        cudaMemcpy(&totalUnusedFlow, d_unusedFlow, sizeof(double), cudaMemcpyDeviceToHost);
        printf("Iteration %d: Total unused flow = %f\n", iteration, totalUnusedFlow);

        redistributionNeeded = totalUnusedFlow > 1e-4;
    }

    if (redistributionNeeded) {
        printf("Finalizing redistribution of unused flow\n");
        for (int finalPass = 0; finalPass < 20; finalPass++) {
            finalizeRedistribution << <blocksPerGrid, threadsPerBlock >> > (d_unitsDelivered, d_demands, d_successRates, d_unusedFlow, size);

            double remainingFlow;
            cudaMemcpy(&remainingFlow, d_unusedFlow, sizeof(double), cudaMemcpyDeviceToHost);
            printf("Final pass %d: Remaining unused flow = %f\n", finalPass + 1, remainingFlow);

            if (remainingFlow < 1e-6) {
                printf("Unused flow fully redistributed after final pass %d\n", finalPass + 1);
                break;
            }
        }

        // Force redistribute remaining flow equally if needed
        double remainingFlow;
        cudaMemcpy(&remainingFlow, d_unusedFlow, sizeof(double), cudaMemcpyDeviceToHost);
        if (remainingFlow > 0) {
            printf("Forcing redistribution of remaining flow\n");
            forceRedistributeRemainingFlow << <blocksPerGrid, threadsPerBlock >> > (d_unitsDelivered, d_demands, d_unusedFlow, size);
        }
    }

    cudaMemcpy(unitsDelivered.data(), d_unitsDelivered, size * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_unitsDelivered);
    cudaFree(d_demands);
    cudaFree(d_successRates);
    cudaFree(d_unusedFlow);
    cudaFree(d_totalUnitsDelivered);
    cudaFree(d_totalDemand);
}

extern "C" void CudaEqualDistributionAlgorithm(NetworkGraph & graph, vector<pair<string, string>> commodities, vector<double> demands) {
    vector<double> unitsDelivered(commodities.size(), 0.0);
    vector<double> successRates(commodities.size(), 0.0);

    vector<int> beforeRedistribution(commodities.size(), 0);

    bool moreFlowNeeded = true;

    while (moreFlowNeeded) {
        moreFlowNeeded = false;

        // Update success rates and find the lowest
        double totalUnitsDelivered = 0, totalDemand = 0;

        CudaRedistributeFlow(graph, commodities, demands, unitsDelivered, successRates);

        for (size_t i = 0; i < commodities.size(); ++i) {
            const string& source = commodities[i].first;
            const string& destination = commodities[i].second;
            double remainingDemand = demands[i] - unitsDelivered[i];

            vector<vector<string>> allPaths = findAllPaths(graph.getEdges(), source, destination);
            sort(allPaths.begin(), allPaths.end(), [&](const vector<string>& a, const vector<string>& b) {
                double weightA = 0, weightB = 0;
                for (size_t j = 0; j < a.size() - 1; ++j) {
                    weightA += graph.getEdge(a[j], a[j + 1]).weight;
                }
                for (size_t j = 0; j < b.size() - 1; ++j) {
                    weightB += graph.getEdge(b[j], b[j + 1]).weight;
                }
                return weightA < weightB;
                });

            for (const auto& path : allPaths) {
                if (remainingDemand <= 0) break;

                double pathCapacity = numeric_limits<double>::max();
                for (size_t j = 0; j < path.size() - 1; ++j) {
                    Edge& edge = graph.getEdge(path[j], path[j + 1]);
                    pathCapacity = min(pathCapacity, static_cast<double>(edge.capacity - edge.flow));
                }

                if (pathCapacity > 0) {
                    double flowToSend = min(remainingDemand, pathCapacity);
                    cudaSendFlow(graph, path, flowToSend);
                    unitsDelivered[i] += flowToSend;
                    remainingDemand -= flowToSend;
                }
            }
        }
    }

    // Capture the results before redistribution
    for (size_t i = 0; i < commodities.size(); ++i) {
        beforeRedistribution[i] = static_cast<int>(round(unitsDelivered[i]));
    }

    // Perform redistribution to ensure fairness
    CudaRedistributeFlow(graph, commodities, demands, unitsDelivered, successRates);

    cout << "\nFinal Results: Units Successfully Reaching Destinations (Before Redistribution)\n";
    for (size_t i = 0; i < commodities.size(); ++i) {
        cout << "Commodity " << i + 1 << " (From " << commodities[i].first
            << " to " << commodities[i].second << "): "
            << beforeRedistribution[i] << "/" << static_cast<int>(demands[i]) << " units\n";
    }

    cout << "\nFinal Results: Units Successfully Reaching Destinations (After Redistribution)\n";
    for (size_t i = 0; i < commodities.size(); ++i) {
        cout << "Commodity " << i + 1 << " (From " << commodities[i].first
            << " to " << commodities[i].second << "): "
            << static_cast<int>(round(unitsDelivered[i])) << "/" << static_cast<int>(demands[i]) << " units\n";
    }

    // Show the difference
    cout << "\nDifferences Between Before and After Redistribution:\n";
    for (size_t i = 0; i < commodities.size(); ++i) {
        int difference = static_cast<int>(round(unitsDelivered[i])) - beforeRedistribution[i];
        cout << "Commodity " << i + 1 << " (From " << commodities[i].first
            << " to " << commodities[i].second << "): "
            << ((difference >= 0) ? "+" : "") << difference << " units\n";
    }
}
