#include <iostream>
#include <vector>
#include <float.h>
#include "NetworkGraph.hpp"
#include "PathFinder.hpp"
#include "Commodity.hpp"

using namespace std;
const double MAX_WEIGHT = 1e9;
const double DECAY = 0.9;

double calculate_path_weight(const Graph& g, const std::vector<int>& path) {
    double total_weight = 0.0;

    for (size_t i = 1; i < path.size(); ++i) {
        auto e = boost::edge(path[i - 1], path[i], g).first;
        total_weight += g[e].weight; // Add the weight of the edge to the total
    }

    return total_weight;
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

void flowDistributionAlgorithm(Graph& g, vector<Commodity>& commodities, double num_of_iter) {
    for (int i = 0; i < num_of_iter; i++) {
        if (i == num_of_iter - 1)
            cout << "Iteration " << i << endl;
        for (int j = 0; j < commodities.size(); j++) {
            // Retrieve all shortest paths for all source-destination pair
            std::vector<std::vector<std::vector<int>>> all_shortest_paths = find_all_shortest_paths(g);

            const std::vector<int>& path = all_shortest_paths[commodities[j].source][commodities[j].destination];

            if (calculate_path_weight(g, path) >= DBL_MAX) {
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

            /*for (auto e : boost::make_iterator_range(boost::edges(g))) {
                auto source_node = boost::source(e, g);
                auto target_node = boost::target(e, g);

                auto flow = g[e].flow;
                auto capacity = g[e].capacity;

                if (g[e].flow != 0) {
                    std::cout << source_node << " -> " << target_node
                        << " [Flow: " << flow << ", Capacity: " << capacity << "]\n";
                }
            }
            cout << "=======================================================\n";*/

            commodities[j].sent += total_flow_assigned; // Update the total sent flow for the commodity
        }
    }
}