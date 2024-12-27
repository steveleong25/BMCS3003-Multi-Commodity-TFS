#include <mpi.h>
#include <cmath>
#include <vector>
#include <iostream>
#include "NetworkGraph.hpp"
#include "PathFinder.hpp"
#include "Commodity.hpp"

using namespace std;

// Broadcast a graph to all processes
void broadcastGraph(Graph& g, int rank) {
    if (rank == 0) {
        // Serialize the graph
        vector<int> edge_data; // To store edge data (source, target, capacity, weight)
        for (auto e : boost::make_iterator_range(boost::edges(g))) {
            int source = boost::source(e, g);
            int target = boost::target(e, g);
            double capacity = g[e].capacity;
            double weight = g[e].weight;

            // Flatten the data into integers and doubles
            edge_data.push_back(source);
            edge_data.push_back(target);
            edge_data.push_back(*(int*)&capacity); // Cast double to int
            edge_data.push_back(*(int*)&weight);  // Cast double to int
        }

        // Broadcast the number of edges
        int num_edges = edge_data.size();
        MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Broadcast the serialized graph data
        MPI_Bcast(edge_data.data(), num_edges, MPI_INT, 0, MPI_COMM_WORLD);
    }
    else {
        // Receive the number of edges
        int num_edges;
        MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Receive the serialized graph data
        vector<int> edge_data(num_edges);
        MPI_Bcast(edge_data.data(), num_edges, MPI_INT, 0, MPI_COMM_WORLD);

        // Deserialize the graph
        for (size_t i = 0; i < edge_data.size(); i += 4) {
            int source = edge_data[i];
            int target = edge_data[i + 1];
            double capacity = *(double*)&edge_data[i + 2]; // Cast back to double
            double weight = *(double*)&edge_data[i + 3];  // Cast back to double

            auto e = boost::add_edge(source, target, g).first;
            g[e].capacity = capacity;
            g[e].weight = weight;
        }
    }
}

// Broadcast a vector of commodities
void broadcastCommodities(vector<Commodity>& commodities, int rank) {
    if (rank == 0) {
        // Serialize the commodities
        vector<double> commodity_data; // To store commodity data (source, destination, demand, sent)
        for (const auto& commodity : commodities) {
            commodity_data.push_back(commodity.source);
            commodity_data.push_back(commodity.destination);
            commodity_data.push_back(commodity.demand);
            commodity_data.push_back(commodity.sent);
        }

        // Broadcast the number of commodities
        int num_commodities = commodity_data.size();
        MPI_Bcast(&num_commodities, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Broadcast the serialized commodities
        MPI_Bcast(commodity_data.data(), num_commodities, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else {
        // Receive the number of commodities
        int num_commodities;
        MPI_Bcast(&num_commodities, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Receive the serialized commodity data
        vector<double> commodity_data(num_commodities);
        MPI_Bcast(commodity_data.data(), num_commodities, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Deserialize the commodities
        commodities.clear();
        for (size_t i = 0; i < commodity_data.size(); i += 4) {
            Commodity commodity;
            commodity.source = commodity_data[i];
            commodity.destination = commodity_data[i + 1];
            commodity.demand = commodity_data[i + 2];
            commodity.sent = commodity_data[i + 3];
            commodities.push_back(commodity);
        }
    }
}

// scatter edges among processes
vector<boost::graph_traits<Graph>::edge_descriptor> scatterEdges(
    const vector<boost::graph_traits<Graph>::edge_descriptor>& edges,
    int rank, int size
) {
    int total_edges = edges.size();
    int edges_per_proc = (total_edges + size - 1) / size;

    int start = rank * edges_per_proc;
    int end = min(start + edges_per_proc, total_edges);

    return vector<boost::graph_traits<Graph>::edge_descriptor>(
        edges.begin() + start, edges.begin() + end
    );
}

double calculate_bottleneck(Graph& g, const vector<boost::graph_traits<Graph>::edge_descriptor>& edges_with_flow) {
    double local_min_ratio = std::numeric_limits<double>::max();

    for (auto e : edges_with_flow) {
        if (g[e].flow > 0) {
            double ratio = (double)g[e].capacity / g[e].flow;
            local_min_ratio = std::min(local_min_ratio, ratio);
        }
    }

    double global_min_ratio;
    MPI_Allreduce(&local_min_ratio, &global_min_ratio, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    return global_min_ratio;
}

void normalize_flows(Graph& g, double bottleneck_value, const vector<boost::graph_traits<Graph>::edge_descriptor>& edges_with_flow) {
    for (auto e : edges_with_flow) {
        if (g[e].flow > 0) {
            g[e].flow *= bottleneck_value;
        }
    }
}

void updateCommoditiesSent(vector<Commodity>& commodities, double bottleneck_value, int rank, int size) {
    for (auto& commodity : commodities) {
        commodity.sent *= bottleneck_value;
    }
    // If global synchronization is needed, use MPI_Allreduce to aggregate results
}

void recalculate_weights(Graph& g, double alpha, const vector<boost::graph_traits<Graph>::edge_descriptor>& edges_with_flow) {
    for (auto e : edges_with_flow) {
        double flow_ratio = g[e].flow / g[e].capacity;
        g[e].weight = std::exp(alpha * flow_ratio); // exponential weight
    }
}

bool isFlowExceedingCapacity(Graph& g, const vector<boost::graph_traits<Graph>::edge_descriptor>& edges_with_flow) {
    bool local_exceeding = false;
    for (auto e : edges_with_flow) {
        if (g[e].flow > g[e].capacity) {
            local_exceeding = true;
            break;
        }
    }

    bool global_exceeding;
    MPI_Allreduce(&local_exceeding, &global_exceeding, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
    return global_exceeding;
}

vector<boost::graph_traits<Graph>::edge_descriptor> get_edges_with_flow(Graph& g) {
    vector<boost::graph_traits<Graph>::edge_descriptor> edges_with_flow;

    for (auto e : boost::make_iterator_range(boost::edges(g))) {
        if (g[e].flow > 0) {
            edges_with_flow.push_back(e);
        }
    }

    return edges_with_flow;
}

// Main MPI flow distribution algorithm
double MPI_flowDistributionAlgorithm(Graph& g, vector<Commodity>& commodities, double epsilon, double alpha, int rank, int size) {
    double solution = 0.0;

    // Step 1: Broadcast graph and commodities to all processes
    if (rank == 0) {
        broadcastGraph(g, rank);
        broadcastCommodities(commodities, rank);
    }
    else {
        broadcastGraph(g, rank);
        broadcastCommodities(commodities, rank);
    }

    // Step 2: Scatter edges_with_flow among processes
    vector<boost::graph_traits<Graph>::edge_descriptor> edges_with_flow = get_edges_with_flow(g);
    vector<boost::graph_traits<Graph>::edge_descriptor> local_edges = scatterEdges(edges_with_flow, rank, size);

    double prev_max_ratio = 0.0;

    while (true) {
        // Step 3: Calculate bottleneck value
        double bottleneck_value = calculate_bottleneck(g, local_edges);

        // Step 4: Normalize flows
        if (isFlowExceedingCapacity(g, local_edges)) {
            normalize_flows(g, bottleneck_value, local_edges);
            updateCommoditiesSent(commodities, bottleneck_value, rank, size);
        }

        // Step 5: Recalculate weights
        recalculate_weights(g, alpha, local_edges);

        // Step 6: Compute local max ratio
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

        // Reduce max ratio across processes
        double global_max_ratio;
        MPI_Allreduce(&local_max_ratio, &global_max_ratio, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

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