#include <iostream>
#include <map>
#include <vector>
#include <omp.h>
#include "NetworkGraph.hpp"
#include "PathFinder.hpp"
#include "Commodity.hpp"

using namespace std;

typedef boost::graph_traits<Graph>::edge_descriptor EdgeDescriptor;

int calculate_bottleneck(const Graph& g, const std::vector<std::vector<int>>& assigned_paths) {
    // track total usage of the edges
    std::map<std::pair<int, int>, int> edge_usage;

    // flow usage count
    for (const auto& path : assigned_paths) {
        for (size_t i = 1; i < path.size(); ++i) {
            int u = path[i - 1];
            int v = path[i];
            edge_usage[{u, v}] += 1;
        }
    }

    EdgeDescriptor bottleneck_edge;
    int max_flow = 0;

    boost::graph_traits<Graph>::edge_iterator ei, ei_end;
    for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei) {
        int u = boost::source(*ei, g);
        int v = boost::target(*ei, g);

        const auto& edge = g[*ei];
        int total_flow = edge_usage[{u, v}];

        if (total_flow == 0) continue;

        if (total_flow > max_flow) {
            max_flow = total_flow;
            bottleneck_edge = *ei;
        }
    }

    return g[bottleneck_edge].capacity;
}

void normalize_flows(Graph& g, std::vector<std::vector<int>>& assigned_paths, double bottleneck_value) {
    for (auto& path : assigned_paths) {
        for (size_t i = 1; i < path.size(); ++i) {
            auto e = boost::edge(path[i - 1], path[i], g).first;
            int edge_capacity = g[e].capacity;
            int edge_flow = g[e].flow;

            // Check if exceeds capacity, normalize only if so
            if (edge_flow > edge_capacity) {
                g[e].flow /= bottleneck_value;
            }
        }
    }
}

void recalculate_weights(Graph& g, double alpha) {
    boost::graph_traits<Graph>::edge_iterator ei, ei_end;
    for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei) {
        auto& edge = g[*ei];
        edge.weight = std::exp(alpha * edge.flow);
    }
}

double flowDistributionAlgorithm(Graph& g, vector<Commodity>& commodities, double epsilon, double alpha) {
    double total_demand = 0.0, solution = 0.0;
    std::vector<std::vector<int>> assigned_paths;

    // Initialize flows and assign initial flows
    for (auto& commodity : commodities) {
        total_demand += commodity.demand;

        std::vector<int> path = find_shortest_path(g, commodity.source, commodity.destination);
        assigned_paths.push_back(path);

        for (size_t i = 1; i < path.size(); ++i) {
            auto e = boost::edge(path[i - 1], path[i], g).first;
            g[e].flow += commodity.demand;
        }
    }

    double prev_max_ratio = 0.0;
    while (true) {
        // Get the bottleneck value 
        double bottleneck_value = calculate_bottleneck(g, assigned_paths);

        // Normalize flows
		normalize_flows(g, assigned_paths, bottleneck_value);

        // Recalculate weights
        recalculate_weights(g, alpha);

        // Compute the max ratio after redistribution
        double max_ratio = 0.0;
        max_ratio = bottleneck_value / total_demand;

        // Check for convergence
        if (std::abs(max_ratio - prev_max_ratio) < epsilon) {
            solution = max_ratio;
            break;
        }
        prev_max_ratio = max_ratio;

        // Reassign flows for the next iteration
		assigned_paths.clear();
        for (auto& commodity : commodities) {
            std::vector<int> path = find_shortest_path(g, commodity.source, commodity.destination);
			assigned_paths.push_back(path);
            for (size_t i = 1; i < path.size(); ++i) {
                auto e = boost::edge(path[i - 1], path[i], g).first;
                g[e].flow += commodity.demand;
            }
        }
    }

    // Calculate fulfilled demand
	cout << "Total demand: " << total_demand << endl;

    return solution;
}