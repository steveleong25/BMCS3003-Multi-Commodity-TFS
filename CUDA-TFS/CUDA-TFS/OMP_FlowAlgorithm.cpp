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
        g[e].weight *= exp(alpha * flow_ratio); // exponential weight 
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
    #pragma omp parallel for
    for (i = 0; i < commodities.size(); i++) {
        std::vector<int> path = find_shortest_path(g, commodities[i].source, commodities[i].destination);
        if (path.empty()) continue;

        for (int j = 1; j < path.size(); ++j) {
            auto e = boost::edge(path[j - 1], path[j], g).first;
            #pragma omp atomic
            g[e].flow += commodities[i].demand;
            commodities[i].sent = g[e].flow;
        }
    }    

    vector<boost::graph_traits<Graph>::edge_descriptor> edges_with_flow = OMP_get_edges_with_flow(g);

    double prev_max_ratio = 0.0;
    while (true) {
        // get bottleneck value
        double bottleneck_value = parallel_calculate_bottleneck(g, edges_with_flow);

        // normalize flow
        if (parallel_isFlowExceedingCapacity(g, edges_with_flow)) {
            parallel_normalize_flows(g, bottleneck_value, edges_with_flow);
            parallel_updateCommoditiesSent(commodities, bottleneck_value);
        }

        for (auto e : edges_with_flow) {
            auto source_node = boost::source(e, g);
            auto target_node = boost::target(e, g);

            auto flow = g[e].flow;
            auto capacity = g[e].capacity;

            if (g[e].flow > 0) {
                std::cout << source_node << " -> " << target_node
                    << " [Flow: " << flow << ", Capacity: " << capacity << "]\n";
            }
        }

        parallel_recalculate_weights(g, alpha, edges_with_flow);

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

                if (total_flow_on_edge > local_highest_flow) {
                    local_highest_flow = total_flow_on_edge;
                    local_min_capacity = edge_capacity;  // reassign min capacity
                    local_max_ratio = edge_capacity / total_flow_on_edge;
                }
                else if (total_flow_on_edge == local_highest_flow) {
                    if (edge_capacity < local_min_capacity) {
                        local_min_capacity = edge_capacity;  
                        local_max_ratio = edge_capacity / total_flow_on_edge;
                    }
                }
            }

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

        if (abs(max_ratio - prev_max_ratio) < epsilon) {
            solution = max_ratio;
            break;
        }
        prev_max_ratio = max_ratio;
    }

    return solution;
}
