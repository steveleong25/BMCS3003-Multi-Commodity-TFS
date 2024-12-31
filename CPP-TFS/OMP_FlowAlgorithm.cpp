#include <iostream>
#include <vector>
#include "NetworkGraph.hpp"
#include "PathFinder.hpp"
#include "Commodity.hpp"
#include <omp.h>

using namespace std;

double parallel_calculate_bottleneck(Graph& g, vector<boost::graph_traits<Graph>::edge_descriptor>& edges_with_flow) {
    double min_ratio = std::numeric_limits<double>::max();

#pragma omp parallel
    {
        double local_min_ratio = std::numeric_limits<double>::max();

        #pragma omp for
        for (int i = 0; i < edges_with_flow.size(); i++) {
            auto e = edges_with_flow[i];

            if (g[e].flow > 0) {
                double ratio = static_cast<double>(g[e].capacity) / g[e].flow;
                local_min_ratio = std::min(local_min_ratio, ratio);
            }
        }

        #pragma omp critical
        {
            min_ratio = std::min(min_ratio, local_min_ratio);
        }
    }

    return min_ratio;
}

void parallel_normalize_flows(Graph& g, double bottleneck_value, vector<boost::graph_traits<Graph>::edge_descriptor>& edges_with_flow) {
    int i;
#pragma omp parallel for
    for (i = 0; i < edges_with_flow.size(); i++) {
        auto e = edges_with_flow[i];
        g[e].flow *= bottleneck_value;
    }
}

void parallel_updateCommoditiesSent(vector<Commodity>& commodities, double bottleneck_value) {
    int i;
#pragma omp parallel for
    for (i = 0; i < commodities.size(); i++) {
        commodities[i].sent *= bottleneck_value;
    }
}

void parallel_recalculate_weights(Graph& g, double alpha, vector<boost::graph_traits<Graph>::edge_descriptor>& edges_with_flow) {
    int i;
#pragma omp parallel for
    for (i = 0; i < edges_with_flow.size(); i++) {
        auto e = edges_with_flow[i];
        double flow_ratio = g[e].flow / g[e].capacity;
        g[e].weight = exp(alpha * flow_ratio); // exponential weight 
    }
}

bool parallel_isFlowExceedingCapacity(Graph& g, vector<boost::graph_traits<Graph>::edge_descriptor>& edges_with_flow) {
    bool result = false;
    int i;

    #pragma omp parallel for
    for (i = 0; i < edges_with_flow.size(); i++) {
        if (result) continue;

        auto e = edges_with_flow[i];
        if (g[e].flow > g[e].capacity) {
            #pragma omp critical
            {
                result = true;
            }
        }
    }
    return result;
}

vector<boost::graph_traits<Graph>::edge_descriptor> OMP_get_edges_with_flow(Graph& g) {
    vector<boost::graph_traits<Graph>::edge_descriptor> edges_with_flow;

    for (auto e : boost::make_iterator_range(boost::edges(g))) {
        if (g[e].flow > 0) {
            edges_with_flow.push_back(e);
        }
    }

    return edges_with_flow;
}

double OMP_flowDistributionAlgorithm(Graph& g, vector<Commodity>& commodities, double epsilon, double alpha) {
    double solution = 0.0;
    int i;

    double prev_max_ratio = 0.0;
    std::vector<std::vector<std::vector<int>>> all_shortest_paths = find_all_shortest_paths(g);

    while (true) {
        #pragma omp parallel for
        for (i = 0; i < commodities.size(); i++) {
            // Retrieve the shortest path for this source-destination pair
            const std::vector<int>& path = all_shortest_paths[commodities[i].source][commodities[i].destination];

            if (path.empty()) {
                std::cerr << "No path exists between source " << commodities[i].source
                    << " and destination " << commodities[i].destination << std::endl;
                continue;
            }

            // Use the path to distribute flows
            for (size_t i = 1; i < path.size(); ++i) {
                auto e = boost::edge(path[i - 1], path[i], g).first;
                g[e].flow += commodities[i].demand;
                commodities[i].sent = g[e].flow;
            }
        }

        vector<boost::graph_traits<Graph>::edge_descriptor> edges_with_flow = OMP_get_edges_with_flow(g);

        // get bottleneck value
        double bottleneck_value = parallel_calculate_bottleneck(g, edges_with_flow);

        // normalize flow
        if (parallel_isFlowExceedingCapacity(g, edges_with_flow)) {
            parallel_normalize_flows(g, bottleneck_value, edges_with_flow);
            parallel_updateCommoditiesSent(commodities, bottleneck_value);
        }

        for (auto c : commodities) {
            c.demand = c.init_demand - c.sent;
        }

        // recalculate weights
        parallel_recalculate_weights(g, alpha, edges_with_flow);

        // compute the max ratio after redistribution       
        double max_ratio = 0.0;
        double highest_flow = 0.0;
        double min_capacity = std::numeric_limits<double>::max();

        #pragma omp parallel
        {
            double local_max_ratio = 0.0;
            double local_highest_flow = 0.0;
            double local_min_capacity = std::numeric_limits<double>::max();

            #pragma omp for
            for (i = 0; i < edges_with_flow.size(); ++i) {
                auto e = edges_with_flow[i];
                if (g[e].flow == 0) continue;  // skip edges with no flow

                double total_flow_on_edge = g[e].flow;
                double edge_capacity = g[e].capacity;

                // if current edge flow is greater than the current highest flow in this thread
                if (total_flow_on_edge > local_highest_flow) {
                    local_highest_flow = total_flow_on_edge;
                    local_min_capacity = edge_capacity;  // reassign min capacity
                    local_max_ratio = edge_capacity / total_flow_on_edge;
                }
                // if current edge flow is equal to the current highest flow, update by min_capacity
                else if (total_flow_on_edge == local_highest_flow) {
                    if (edge_capacity < local_min_capacity) {
                        local_min_capacity = edge_capacity;
                        local_max_ratio = edge_capacity / total_flow_on_edge;
                    }
                }
            }

            // update global max_ratio, highest_flow, and min_capacity
            #pragma omp critical
            {
                if (local_highest_flow > highest_flow) {
                    highest_flow = local_highest_flow;
                    min_capacity = local_min_capacity;
                    max_ratio = local_max_ratio;
                }
                else if (local_highest_flow == highest_flow) {
                    if (local_min_capacity < min_capacity) {
                        min_capacity = local_min_capacity;
                        max_ratio = local_max_ratio;
                    }
                }
            }
        }

        bool allFulfilled = true;
        for (auto& commodity : commodities) {
            if (commodity.sent != commodity.demand) {
                allFulfilled = false;
                break;
            }
        }

        // convergence
        if (abs(max_ratio - prev_max_ratio) < epsilon || allFulfilled) {
            solution = max_ratio;
            break;
        }
        prev_max_ratio = max_ratio;
    }

    return solution;
}