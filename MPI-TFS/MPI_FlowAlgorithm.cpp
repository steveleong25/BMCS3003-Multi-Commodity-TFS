#include <mpi.h>
#include "NetworkGraph.hpp"
#include "PathFinder.hpp"
#include "Commodity.hpp"
#include "MPI_FlowAlgorithm.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/random.hpp>
#include <boost/random.hpp>
#include <iostream>

using namespace std;

double MPI_calculate_bottleneck(Graph& g, std::vector<boost::graph_traits<Graph>::edge_descriptor>& edges_with_flow) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double local_min_ratio = std::numeric_limits<double>::max();

    // Divide edges among processes
    int total_edges = edges_with_flow.size();
    int chunk_size = (total_edges + size - 1) / size; // Calculate the chunk size per process
    int start = rank * chunk_size;
    int end = std::min(start + chunk_size, total_edges);

    // Each process calculates its local minimum
    for (int i = start; i < end; ++i) {
        auto e = edges_with_flow[i];
        if (g[e].flow > 0) {
            double ratio = static_cast<double>(g[e].capacity) / g[e].flow;
            local_min_ratio = std::min(local_min_ratio, ratio);
        }
    }

    // Reduce the local minima to the global minimum
    double global_min_ratio;
    MPI_Allreduce(&local_min_ratio, &global_min_ratio, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    return global_min_ratio;
}

void MPI_normalize_flows(Graph& g, double bottleneck_value, std::vector<boost::graph_traits<Graph>::edge_descriptor>& edges_with_flow) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide edges among processes
    int total_edges = edges_with_flow.size();
    int chunk_size = (total_edges + size - 1) / size; // Calculate the chunk size per process
    int start = rank * chunk_size;
    int end = std::min(start + chunk_size, total_edges);

    // Each process updates its assigned edges
    for (int i = start; i < end; ++i) {
        auto e = edges_with_flow[i];
        g[e].flow *= bottleneck_value;
    }

    // Synchronize graph updates across processes
    // Assuming the graph object is shared and synchronized; if not, an alternative approach like graph partitioning is needed.
}

void MPI_updateCommoditiesSent(std::vector<Commodity>& commodities, double bottleneck_value) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide commodities among processes
    int total_commodities = commodities.size();
    int chunk_size = (total_commodities + size - 1) / size; // Calculate the chunk size per process
    int start = rank * chunk_size;
    int end = std::min(start + chunk_size, total_commodities);

    // Each process updates its assigned commodities
    for (int i = start; i < end; ++i) {
        commodities[i].sent *= bottleneck_value;
    }

    // Synchronize updates (if necessary)
    // Gather updated commodities data at root process or all processes
    std::vector<Commodity> updated_commodities;
    if (rank == 0) {
        updated_commodities.resize(total_commodities);
    }

    // Assuming Commodity is a plain struct and can be directly communicated
    MPI_Gather(
        commodities.data() + start, chunk_size * sizeof(Commodity), MPI_BYTE,
        updated_commodities.data(), chunk_size * sizeof(Commodity), MPI_BYTE,
        0, MPI_COMM_WORLD
    );

    // Broadcast updated commodities to all processes if needed
    MPI_Bcast(updated_commodities.data(), total_commodities * sizeof(Commodity), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        commodities = updated_commodities;
    }
}

void MPI_recalculate_weights(Graph& g, double alpha, std::vector<boost::graph_traits<Graph>::edge_descriptor>& edges_with_flow) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide edges among processes
    int total_edges = edges_with_flow.size();
    int chunk_size = (total_edges + size - 1) / size; // Calculate the chunk size per process
    int start = rank * chunk_size;
    int end = std::min(start + chunk_size, total_edges);

    // Each process recalculates weights for its assigned edges
    for (int i = start; i < end; ++i) {
        auto e = edges_with_flow[i];
        double flow_ratio = g[e].flow / g[e].capacity;
        g[e].weight = std::exp(alpha * flow_ratio); // Exponential weight update
    }

    // Synchronize updates across all processes (if necessary)
    // Collect updated weights from all processes (optional)
    // Assuming `g` is a shared or distributed structure and needs global consistency
    std::vector<double> local_weights(chunk_size);
    for (int i = 0; i < chunk_size && start + i < total_edges; ++i) {
        auto e = edges_with_flow[start + i];
        local_weights[i] = g[e].weight;
    }

    std::vector<double> global_weights(total_edges);
    MPI_Gather(
        local_weights.data(), chunk_size, MPI_DOUBLE,
        global_weights.data(), chunk_size, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    if (rank == 0) {
        // Update the weights in the graph globally (optional)
        for (int i = 0; i < total_edges; ++i) {
            auto e = edges_with_flow[i];
            g[e].weight = global_weights[i];
        }
    }

    // Broadcast the updated graph weights to all processes
    MPI_Bcast(global_weights.data(), total_edges, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Update local graph weights after broadcasting
    for (int i = 0; i < total_edges; ++i) {
        auto e = edges_with_flow[i];
        g[e].weight = global_weights[i];
    }
}


bool MPI_isFlowExceedingCapacity(Graph& g, std::vector<boost::graph_traits<Graph>::edge_descriptor>& edges_with_flow) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide edges among processes
    int total_edges = edges_with_flow.size();
    int chunk_size = (total_edges + size - 1) / size; // Calculate the chunk size per process
    int start = rank * chunk_size;
    int end = std::min(start + chunk_size, total_edges);

    // Local result
    bool local_result = false;

    // Check if flow exceeds capacity for assigned edges
    for (int i = start; i < end; ++i) {
        if (local_result) break; // Early exit if true is found locally
        auto e = edges_with_flow[i];
        if (g[e].flow > g[e].capacity) {
            local_result = true;
        }
    }

    // Global reduction to determine if any process found a flow exceeding capacity
    bool global_result = false;
    MPI_Allreduce(&local_result, &global_result, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);

    return global_result;
}

std::vector<boost::graph_traits<Graph>::edge_descriptor> MPI_get_edges_with_flow(Graph& g) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get all edges in the graph
    auto edge_range = boost::edges(g);
    int total_edges = boost::num_edges(g);
    int chunk_size = (total_edges + size - 1) / size; // Calculate the chunk size per process
    int start = rank * chunk_size;
    int end = std::min(start + chunk_size, total_edges);

    // Iterate only over the edges assigned to this process
    std::vector<boost::graph_traits<Graph>::edge_descriptor> local_edges_with_flow;
    auto edge_iter = edge_range.first;

    std::advance(edge_iter, start); // Move to the starting edge for this process

    for (int i = start; i < end && edge_iter != edge_range.second; ++i, ++edge_iter) {
        auto e = *edge_iter;
        if (g[e].flow > 0) {
            local_edges_with_flow.push_back(e);
        }
    }

    // Gather results from all processes
    int local_size = local_edges_with_flow.size();
    std::vector<int> all_sizes(size);

    MPI_Allgather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Calculate displacements for gathered data
    std::vector<int> displacements(size, 0);
    for (int i = 1; i < size; ++i) {
        displacements[i] = displacements[i - 1] + all_sizes[i - 1];
    }

    int total_size = displacements[size - 1] + all_sizes[size - 1];
    std::vector<boost::graph_traits<Graph>::edge_descriptor> global_edges_with_flow(total_size);

    MPI_Allgatherv(local_edges_with_flow.data(), local_size, MPI_BYTE,
        global_edges_with_flow.data(), all_sizes.data(), displacements.data(), MPI_BYTE,
        MPI_COMM_WORLD);

    return global_edges_with_flow;
}

double MPI_flowDistributionAlgorithm(Graph& g, std::vector<Commodity>& commodities, double epsilon, double alpha) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double solution = 0.0;

    // Distribute commodities among processes
    int total_commodities = commodities.size();
    int chunk_size = (total_commodities + size - 1) / size;
    int start = rank * chunk_size;
    int end = std::min(start + chunk_size, total_commodities);

    // Step 1: Each process finds paths and updates flow for its assigned commodities
    for (int i = start; i < end; ++i) {
        std::vector<int> path = find_shortest_path(g, commodities[i].source, commodities[i].destination);
        if (path.empty()) continue;

        for (size_t j = 1; j < path.size(); ++j) {
            auto e = boost::edge(path[j - 1], path[j], g).first;
            g[e].flow += commodities[i].demand;
            commodities[i].sent = g[e].flow;
        }
    }

    // Gather all flows and synchronize the graph across processes
    // Assuming g is a shared graph; if it's distributed, consider an approach like graph partitioning
    std::vector<boost::graph_traits<Graph>::edge_descriptor> edges_with_flow = MPI_get_edges_with_flow(g);

    double prev_max_ratio = 0.0;
    while (true) {
        // Step 2: Calculate bottleneck value globally
        double local_bottleneck = MPI_calculate_bottleneck(g, edges_with_flow);
        double global_bottleneck;
        MPI_Allreduce(&local_bottleneck, &global_bottleneck, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        // Step 3: Normalize flows if needed
        bool local_exceeds_capacity = MPI_isFlowExceedingCapacity(g, edges_with_flow);
        bool global_exceeds_capacity;
        MPI_Allreduce(&local_exceeds_capacity, &global_exceeds_capacity, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);

        if (global_exceeds_capacity) {
            MPI_normalize_flows(g, global_bottleneck, edges_with_flow);
            MPI_updateCommoditiesSent(commodities, global_bottleneck);
        }

        // Step 4: Recalculate weights
        MPI_recalculate_weights(g, alpha, edges_with_flow);

        // Step 5: Compute max ratio
        double local_max_ratio = 0.0, global_max_ratio = 0.0;
        for (const auto& e : edges_with_flow) {
            if (g[e].flow > 0) {
                double ratio = g[e].capacity / g[e].flow;
                local_max_ratio = std::max(local_max_ratio, ratio);
            }
        }
        MPI_Allreduce(&local_max_ratio, &global_max_ratio, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // Step 6: Check for convergence
        if (std::abs(global_max_ratio - prev_max_ratio) < epsilon) {
            solution = global_max_ratio;
            break;
        }
        prev_max_ratio = global_max_ratio;
    }


    return solution;
}

