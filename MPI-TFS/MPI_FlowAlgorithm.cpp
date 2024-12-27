#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>
#include "NetworkGraph.hpp"
#include "PathFinder.hpp"
#include "Commodity.hpp"

using namespace std;
const int INITIAL_WEIGHT = 5;

double calculate_bottleneck(Graph& g, vector<boost::graph_traits<Graph>::edge_descriptor>& edges_with_flow) {
    double local_min_ratio = std::numeric_limits<double>::max();
    for (auto e : edges_with_flow) {
        if (g[e].flow > 0) {
            double ratio = (double)g[e].capacity / g[e].flow;
            local_min_ratio = std::min(local_min_ratio, ratio);
        }
    }
    return local_min_ratio;
}

void normalize_flows(Graph& g, double bottleneck_value, vector<boost::graph_traits<Graph>::edge_descriptor>& edges_with_flow) {
    for (auto e : edges_with_flow) {
        if (g[e].flow > 0) {
            g[e].flow *= bottleneck_value;
        }
    }
}

void updateCommoditiesSent(vector<Commodity>& commodities, double bottleneck_value) {
    for (auto& commodity : commodities) {
        commodity.sent *= bottleneck_value;
    }
}

void updateCommoditiesDemand(vector<Commodity>& commodities) {
    for (auto& commodity : commodities) {
        commodity.demand = commodity.init_demand - commodity.sent;
    }
}

void recalculate_weights(Graph& g, double alpha, vector<boost::graph_traits<Graph>::edge_descriptor>& edges_with_flow) {
    for (auto e : edges_with_flow) {
        g[e].weight = std::min(INITIAL_WEIGHT * (int)ceil(std::exp(alpha * g[e].flow)), INT_MAX);
    }
}

bool isFlowExceedingCapacity(Graph& g, vector<boost::graph_traits<Graph>::edge_descriptor> edges_with_flow) {
    for (auto e : edges_with_flow) {
        if (g[e].flow > g[e].capacity) {
            return true;
        }
    }
    return false;
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

struct PathFlow {
    vector<int> path;
    double flow;
};

double MPI_flowDistributionAlgorithm(Graph& g, vector<Commodity>& commodities, double epsilon, double alpha, int rank, int size) {
    double solution = 0.0;
    double prev_max_ratio = 0.0;

    // Total number of commodities
    int total_commodities = commodities.size();

    // Scatter the commodities
    int chunk_size = total_commodities / size;
    int remainder = total_commodities % size;

    int local_commodities_count = chunk_size + (rank < remainder ? 1 : 0);

    vector<Commodity> local_commodities(local_commodities_count);

    int pack_size = local_commodities_count * sizeof(Commodity);
    char* send_buffer = new char[total_commodities * sizeof(Commodity)];
    char* recv_buffer = new char[pack_size];

    // Pack data into a buffer
    if (rank == 0) {
        int offset = 0;
        for (int i = 0; i < size; ++i) {
            int start_idx = i * chunk_size + std::min(i, remainder);  // starting index for this process, take remainder as the last 
            int end_idx = start_idx + chunk_size + (i < remainder ? 1 : 0); 

            // pack the commodities for the current process
            for (int j = start_idx; j < end_idx; ++j) {
                cout << commodities[j].source << " -> " << commodities[j].destination << endl;
                MPI_Pack(&commodities[j], 1, MPI_BYTE, send_buffer, total_commodities * sizeof(Commodity), &offset, MPI_COMM_WORLD);
            }
        }
    }

    // scatter
    MPI_Scatter(send_buffer, pack_size, MPI_BYTE, recv_buffer, pack_size, MPI_BYTE, 0, MPI_COMM_WORLD);

    // unpack
    MPI_Unpack(recv_buffer, pack_size, &pack_size, local_commodities.data(), local_commodities_count, MPI_BYTE, MPI_COMM_WORLD);

    delete[] send_buffer;
    delete[] recv_buffer;

    while (true) {
        for (auto& commodity : local_commodities) {
            std::vector<int> path = find_shortest_path(g, commodity.source, commodity.destination);

            for (size_t j = 1; j < path.size(); ++j) {
                auto e = boost::edge(path[j - 1], path[j], g).first;
                g[e].flow += commodity.demand;
                commodity.sent = commodity.demand;
            }
        }

        vector<boost::graph_traits<Graph>::edge_descriptor> edges_with_flow = get_edges_with_flow(g);
        double bottleneck_value = calculate_bottleneck(g, edges_with_flow);

        if (isFlowExceedingCapacity(g, edges_with_flow)) {
            normalize_flows(g, bottleneck_value, edges_with_flow);
            updateCommoditiesSent(local_commodities, bottleneck_value);
        }

        updateCommoditiesDemand(local_commodities);

        recalculate_weights(g, alpha, edges_with_flow);

        double max_ratio = 0.0;
        double highest_flow = 0.0, min_capacity = 0.0;

        for (auto e : edges_with_flow) {
            double total_flow_on_edge = g[e].flow;
            double edge_capacity = g[e].capacity;

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

        double global_max_ratio;
        MPI_Reduce(&max_ratio, &global_max_ratio, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (std::abs(global_max_ratio - prev_max_ratio) < epsilon) {
                solution = global_max_ratio;
                break;
            }
            prev_max_ratio = global_max_ratio;
        }

        MPI_Bcast(&prev_max_ratio, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    MPI_Gather(local_commodities.data(), local_commodities_count * sizeof(Commodity), MPI_BYTE,
        commodities.data(), chunk_size * sizeof(Commodity), MPI_BYTE, 0, MPI_COMM_WORLD);

    return solution;
}