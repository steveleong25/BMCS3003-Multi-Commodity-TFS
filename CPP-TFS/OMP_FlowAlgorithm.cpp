#include <iostream>
#include <vector>
#include "NetworkGraph.hpp"
#include "PathFinder.hpp"
#include "Commodity.hpp"
#include <omp.h>

using namespace std;

double parallel_calculate_bottleneck(Graph& g) {
    double min_ratio = std::numeric_limits<double>::max();

    #pragma omp parallel
    {
        double local_min_ratio = std::numeric_limits<double>::max();

        #pragma omp for
        for (int i = 0; i < boost::num_edges(g); i++) {
            auto edge_iter = boost::edges(g);
            std::advance(edge_iter.first, i);
            auto e = *edge_iter.first; // Access the edge

            if (g[e].flow > 0) { // Only consider edges with flow
                double ratio = (double)g[e].capacity / g[e].flow;
                local_min_ratio = std::min(local_min_ratio, ratio);
                
            }
        }

        #pragma omp critical
        {
            min_ratio = std::min(min_ratio, local_min_ratio); // Combine results from all threads
        }
    }

    return min_ratio;
}

void parallel_normalize_flows(Graph& g, double bottleneck_value) {
    #pragma omp parallel for
    for (int i = 0; i < boost::num_edges(g); i++) {
        auto edge_iter = boost::edges(g);
        std::advance(edge_iter.first, i);
        auto e = *edge_iter.first; // Access the edge
        if (g[e].flow > 0) { 
            g[e].flow *= bottleneck_value;
        }
    }
}

void parallel_updateCommoditiesSent(vector<Commodity>& commodities, double bottleneck_value) {
    #pragma omp parallel for
    for (int i = 0; i < commodities.size(); i++) {
        commodities[i].sent *= bottleneck_value;
    }
}

void parallel_recalculate_weights(Graph& g, double alpha) {
    #pragma omp parallel for
    for (int i = 0; i < boost::num_edges(g); i++) {
        auto edge_iter = boost::edges(g);
        std::advance(edge_iter.first, i);
        auto e = *edge_iter.first; // Access the edge
        double flow_ratio = g[e].flow / g[e].capacity;
        g[e].weight = exp(alpha * flow_ratio); // exponential weight 
    }
}

bool parallel_isFlowExceedingCapacity(Graph& g) {
    bool result = false;

    #pragma omp parallel for
    for (int i = 0; i < boost::num_edges(g); i++) {
        auto edge_iter = boost::edges(g);
        std::advance(edge_iter.first, i);
        auto e = *edge_iter.first; // Access the edge
        if (g[e].flow > g[e].capacity) {
            #pragma omp critical
            {
                result = true;
            }
        }
    }
    return result;
}

double OMP_flowDistributionAlgorithm(Graph& g, vector<Commodity>& commodities, double epsilon, double alpha) {
    double solution = 0.0;

    #pragma omp parallel for 
    for (int i = 0; i < commodities.size(); i++) {
        std::vector<int> path = find_shortest_path(g, commodities[i].source, commodities[i].destination);

        for (int j = 1; j < path.size(); ++j) {
            auto e = boost::edge(path[j - 1], path[j], g).first;
        #pragma omp atomic
            g[e].flow += commodities[i].demand;
            commodities[i].sent = g[e].flow;
        }
    }

    double prev_max_ratio = 0.0;
    while (true) {
        // get bottleneck value
        double bottleneck_value = parallel_calculate_bottleneck(g);
        cout << "Bottleneck value: " << bottleneck_value << endl;

        // normalize flow
        if (parallel_isFlowExceedingCapacity(g)) {
            parallel_normalize_flows(g, bottleneck_value);
            parallel_updateCommoditiesSent(commodities, bottleneck_value);
        }

        for (auto e : boost::make_iterator_range(boost::edges(g))) {
            auto source_node = boost::source(e, g);
            auto target_node = boost::target(e, g);

            // get edge properties
            auto flow = g[e].flow;
            auto capacity = g[e].capacity;

            std::cout << source_node << " -> " << target_node
                << " [flow: " << flow << ", capacity: " << capacity << "]\n";
        }

        // recalculate weights
        parallel_recalculate_weights(g, alpha);

        // compute the max ratio after redistribution       
        vector<boost::graph_traits<Graph>::edge_descriptor> edges;
        for (auto e : boost::make_iterator_range(boost::edges(g))) {
            edges.push_back(e);
        }

        double max_ratio = 0.0;
        double highest_flow = 0.0;

        #pragma omp parallel
        {
            double local_max_ratio = 0.0;
            double local_highest_flow = 0.0;

            #pragma omp for
            for (int i = 0; i < edges.size(); ++i) {
                auto e = edges[i];
                double total_flow_on_edge = g[e].flow; 
                double edge_capacity = g[e].capacity;  

                // if this edge.flow > current highest flow in this thread
                if (total_flow_on_edge > local_highest_flow) {
                    local_highest_flow = total_flow_on_edge;
                    local_max_ratio = edge_capacity / total_flow_on_edge;
                }
            }

            // update global max_ratio and highest_flow
            #pragma omp critical
            {
                if (local_highest_flow > highest_flow) {
                    highest_flow = local_highest_flow;
                    max_ratio = local_max_ratio;
                }
            }
        }


        // convergence
        if (abs(max_ratio - prev_max_ratio) < epsilon) {
            solution = max_ratio;
            break;
        }
        prev_max_ratio = max_ratio;
    }

    return solution;
}