#include <mpi.h>
#include <cmath>
#include <vector>
#include <iostream>
#include "NetworkGraph.hpp"
#include "PathFinder.hpp"
#include "Commodity.hpp"

using namespace std;

double flowDistributionAlgorithmMPI(Graph& g, vector<Commodity>& commodities, double epsilon, double alpha, int rank, int size) {
    double solution = 0.0;

    // Step 1: Broadcast graph and commodities to all processes
    if (rank == 0) {
        // Broadcast graph and commodities to all processes (serialization may be required)
        // Example: MPI_Bcast(&g, sizeof(Graph), MPI_BYTE, 0, MPI_COMM_WORLD);
        // Example: MPI_Bcast(&commodities[0], commodities.size() * sizeof(Commodity), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        // Receive graph and commodities
    }

    // Step 2: Divide edges_with_flow among processes
    vector<boost::graph_traits<Graph>::edge_descriptor> edges_with_flow = get_edges_with_flow(g);
    int total_edges = edges_with_flow.size();
    int edges_per_proc = (total_edges + size - 1) / size;

    int start = rank * edges_per_proc;
    int end = min(start + edges_per_proc, total_edges);

    vector<boost::graph_traits<Graph>::edge_descriptor> local_edges(edges_with_flow.begin() + start, edges_with_flow.begin() + end);

    double prev_max_ratio = 0.0;

    while (true) {
        // Step 3: Calculate local bottleneck value
        double local_bottleneck = calculate_bottleneck(g, local_edges);

        // Reduce bottleneck value across all processes
        double bottleneck_value;
        MPI_Allreduce(&local_bottleneck, &bottleneck_value, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        // Normalize flows locally
        if (isFlowExceedingCapacity(g, local_edges)) {
            normalize_flows(g, bottleneck_value, local_edges);
        }

        // Step 4: Recalculate weights locally
        recalculate_weights(g, alpha, local_edges);

        // Step 5: Compute local max ratio
        double local_max_ratio = 0.0;
        double local_highest_flow = 0.0;
        double local_min_capacity = std::numeric_limits<double>::max();

        for (auto e : local_edges) {
            double total_flow_on_edge = g[e].flow;
            double edge_capacity = g[e].capacity;

            if (total_flow_on_edge > local_highest_flow) {
                local_highest_flow = total_flow_on_edge;
                local_min_capacity = edge_capacity;
                local_max_ratio = edge_capacity / total_flow_on_edge;
            }
            else if (total_flow_on_edge == local_highest_flow) {
                if (edge_capacity < local_min_capacity) {
                    local_min_capacity = edge_capacity;
                    local_max_ratio = edge_capacity / total_flow_on_edge;
                }
            }
        }

        // Reduce max ratio, highest flow, and min capacity across all processes
        double global_max_ratio, global_highest_flow, global_min_capacity;
        MPI_Allreduce(&local_max_ratio, &global_max_ratio, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&local_highest_flow, &global_highest_flow, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&local_min_capacity, &global_min_capacity, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        // Check convergence
        if (rank == 0) {
            if (std::abs(global_max_ratio - prev_max_ratio) < epsilon) {
                solution = global_max_ratio;
                break;
            }
            prev_max_ratio = global_max_ratio;
        }

        // Broadcast updated prev_max_ratio to all processes
        MPI_Bcast(&prev_max_ratio, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    return solution;
}