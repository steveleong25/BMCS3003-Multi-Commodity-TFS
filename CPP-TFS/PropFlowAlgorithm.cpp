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

int OMP_calculate_bottleneck(const Graph& g, const std::vector<std::vector<int>>& assigned_paths) {
    // Track total usage of the edges
    std::map<std::pair<int, int>, int> edge_usage;

    // Parallelize edge usage calculation
#pragma omp parallel
    {
        std::map<std::pair<int, int>, int> local_edge_usage; // Thread-local edge usage map

#pragma omp for nowait
        for (int i = 0; i < assigned_paths.size(); ++i) {
            const auto& path = assigned_paths[i];
            for (size_t j = 1; j < path.size(); ++j) {
                int u = path[j - 1];
                int v = path[j];
                local_edge_usage[{u, v}] += 1; // Update thread-local map
            }
        }

        // Merge local maps into the global edge usage map
#pragma omp critical
        for (const auto& entry : local_edge_usage) {
            edge_usage[entry.first] += entry.second;
        }
    }

    // Initialize bottleneck variables
    EdgeDescriptor bottleneck_edge;
    int max_flow = 0;

    // Parallelize bottleneck calculation
    #pragma omp parallel
    {
        int local_max_flow = 0;
        EdgeDescriptor local_bottleneck_edge;

        boost::graph_traits<Graph>::edge_iterator ei, ei_end;
        for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei) {
            int u = boost::source(*ei, g);
            int v = boost::target(*ei, g);

            const auto& edge = g[*ei];
            int total_flow = edge_usage[{u, v}];

            if (total_flow == 0) continue;

            if (total_flow > local_max_flow) {
                local_max_flow = total_flow;
                local_bottleneck_edge = *ei;
            }
        }

        // Reduce results to find the global bottleneck
        #pragma omp critical
        {
            if (local_max_flow > max_flow) {
                max_flow = local_max_flow;
                bottleneck_edge = local_bottleneck_edge;
            }
        }
    }

    return g[bottleneck_edge].capacity;
}

void OMP_normalize_flows(Graph& g, std::vector<std::vector<int>>& assigned_paths, double bottleneck_value) {
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
void OMP_recalculate_weights(Graph& g, double alpha) {
    boost::graph_traits<Graph>::edge_iterator ei, ei_end;

    // Retrieve all edges and their descriptors
    std::vector<boost::graph_traits<Graph>::edge_descriptor> edges;
    for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei) {
        edges.push_back(*ei);
    }

    // Parallelize edge weight recalculation
    #pragma omp parallel for
    for (int i = 0; i < edges.size(); ++i) {
        auto& edge = g[edges[i]];
        edge.weight = std::exp(alpha * edge.flow);
    }
}

double OMP_flowDistributionAlgorithm(Graph& g, vector<Commodity>& commodities, double epsilon, double alpha) {
    double total_demand = 0.0, solution = 0.0;
    std::vector<std::vector<int>> assigned_paths(commodities.size());

    // Initialize flows and assign initial flows
    #pragma omp parallel for reduction(+:total_demand)
    for (int i = 0; i < static_cast<int>(commodities.size()); ++i) {
        total_demand += commodities[i].demand;

        // Find the shortest path for each commodity
        std::vector<int> path = find_shortest_path(g, commodities[i].source, commodities[i].destination);
        assigned_paths[i] = path;

        // Update flows along the path
        for (size_t j = 1; j < path.size(); ++j) {
            auto e = boost::edge(path[j - 1], path[j], g).first;

            #pragma omp atomic
            g[e].flow += commodities[i].demand;
        }
    }

    double prev_max_ratio = 0.0;
    while (true) {
        // Get the bottleneck value
        double bottleneck_value = OMP_calculate_bottleneck(g, assigned_paths);

        // Normalize flows in parallel
        #pragma omp parallel for
        for (int i = 0; i < assigned_paths.size(); ++i) {
            for (size_t j = 1; j < assigned_paths[i].size(); ++j) {
                auto e = boost::edge(assigned_paths[i][j - 1], assigned_paths[i][j], g).first;
                int edge_capacity = g[e].capacity;
                int edge_flow = g[e].flow;

                if (edge_flow > edge_capacity) {
                #pragma omp critical
                    g[e].flow /= bottleneck_value;
                }
            }
        }

        // Recalculate weights in parallel
        OMP_recalculate_weights(g, alpha);

        // Compute the max ratio after redistribution
        double max_ratio = bottleneck_value / total_demand;

        // Check for convergence
        if (std::abs(max_ratio - prev_max_ratio) < epsilon) {
            solution = max_ratio;
            break;
        }
        prev_max_ratio = max_ratio;

        // Reassign flows for the next iteration
        assigned_paths.clear();
        assigned_paths.resize(commodities.size());

#pragma omp parallel for
        for (int i = 0; i < commodities.size(); ++i) {
            std::vector<int> path = find_shortest_path(g, commodities[i].source, commodities[i].destination);
            assigned_paths[i] = path;

            for (size_t j = 1; j < path.size(); ++j) {
                auto e = boost::edge(path[j - 1], path[j], g).first;

#pragma omp atomic
                g[e].flow += commodities[i].demand;
            }
        }
    }

    // Output total demand
    cout << "Total demand: " << total_demand << endl;

    return solution;
}