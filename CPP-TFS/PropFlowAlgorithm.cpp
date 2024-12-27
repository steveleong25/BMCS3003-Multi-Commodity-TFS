#include <iostream>
#include <vector>
#include "NetworkGraph.hpp"
#include "PathFinder.hpp"
#include "Commodity.hpp"

using namespace std;
const int INIT_WEIGHT = 5;

double calculate_bottleneck(Graph& g, vector<boost::graph_traits<Graph>::edge_descriptor> edges_with_flow) {
    double min_ratio = std::numeric_limits<double>::max();
    for (auto e : edges_with_flow) {
        if (g[e].flow > 0) {
            double ratio = (double)g[e].capacity / g[e].flow;
            min_ratio = std::min(min_ratio, ratio);
        }
    }
    return min_ratio;
}

void normalize_flows(Graph& g, double bottleneck_value, vector<boost::graph_traits<Graph>::edge_descriptor> edges_with_flow) {
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

void recalculate_weights(Graph& g, double alpha, vector<boost::graph_traits<Graph>::edge_descriptor> edges_with_flow) {
    for (auto e : edges_with_flow) {
        g[e].weight = std::min(INIT_WEIGHT * (int)ceil(std::exp(alpha * g[e].flow)), INT_MAX); // exponential weight
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

void updateCommoditiesDemand(vector<Commodity>& commodities) {
    for (auto& c : commodities) {
        c.demand = c.init_demand - c.sent;
    }
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

double flowDistributionAlgorithm(Graph& g, vector<Commodity>& commodities, double epsilon, double alpha) {
    double solution = 0.0;

    double prev_max_ratio = 0.0;
    while (true) {
        for (auto& commodity : commodities) {
            std::vector<int> path = find_shortest_path(g, commodity.source, commodity.destination);

            for (size_t i = 1; i < path.size(); ++i) {
                auto e = boost::edge(path[i - 1], path[i], g).first;
                g[e].flow += commodity.demand;
                commodity.sent = g[e].flow;
            }
        }

        vector<boost::graph_traits<Graph>::edge_descriptor> edges_with_flow = get_edges_with_flow(g);


        // calculate the bottleneck value
        double bottleneck_value = calculate_bottleneck(g, edges_with_flow);

        // normalize flows 
        if (isFlowExceedingCapacity(g, edges_with_flow)) {
            normalize_flows(g, bottleneck_value, edges_with_flow);
            updateCommoditiesSent(commodities, bottleneck_value);
        }

        updateCommoditiesDemand(commodities);

        for (auto e : edges_with_flow) {
            auto source_node = boost::source(e, g);
            auto target_node = boost::target(e, g);

            auto flow = g[e].flow;
            auto capacity = g[e].capacity;
        }

        // recalculate weights
        recalculate_weights(g, alpha, edges_with_flow);

        // compute the maximum ratio after redistribution
        double max_ratio = 0.0;
        double highest_flow = 0.0, min_capacity = 0.0;
        boost::graph_traits<Graph>::edge_iterator ei, ei_end;

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

        bool allFulfilled = true;
        for (auto& commodity : commodities) {
            if (commodity.sent != commodity.demand) {
                allFulfilled = false;
                break;
            }
        }

        // convergence
        if (std::abs(max_ratio - prev_max_ratio) < epsilon || allFulfilled) {
            solution = max_ratio;
            break;
        }
        prev_max_ratio = max_ratio;
    }

    return solution;
}