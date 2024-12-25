#include <iostream>
#include <vector>
#include "NetworkGraph.hpp"
#include "PathFinder.hpp"
#include "Commodity.hpp"

using namespace std;

double calculate_bottleneck(Graph& g) {
    double min_ratio = std::numeric_limits<double>::max();
    for (auto e : boost::make_iterator_range(boost::edges(g))) {
        if (g[e].flow > 0) { // Only consider edges with flow
            //cout << "Capacity: " << g[e].capacity << " Flow: " << g[e].flow << endl;
            double ratio = (double)g[e].capacity / g[e].flow;
            min_ratio = std::min(min_ratio, ratio);
        }
    }
    return min_ratio;
}

void normalize_flows(Graph& g, double bottleneck_value) {
    for (auto e : boost::make_iterator_range(boost::edges(g))) {
        if (g[e].flow > 0) { // Scale flows only on edges with flow
            g[e].flow *= bottleneck_value;
        }
    }
}

void updateCommoditiesSent(vector<Commodity>& commodities, double bottleneck_value) {
	for (auto& commodity : commodities) {
	    commodity.sent *= bottleneck_value;
	}
}

void recalculate_weights(Graph& g, double alpha) {
    for (auto e : boost::make_iterator_range(boost::edges(g))) {
        double flow_ratio = g[e].flow / g[e].capacity;
        g[e].weight = std::exp(alpha * flow_ratio); // Exponential weight update
    }
}

bool isFlowExceedingCapacity(Graph& g) {
	for (auto e : boost::make_iterator_range(boost::edges(g))) {
		if (g[e].flow > g[e].capacity) {
			return true;
		}
	}
	return false;
}

double flowDistributionAlgorithm(Graph& g, vector<Commodity>& commodities, double epsilon, double alpha) {
    double solution = 0.0;
    //std::vector<std::vector<int>> assigned_paths;

    for (auto& commodity : commodities) {
        std::vector<int> path = find_shortest_path(g, commodity.source, commodity.destination);
        //assigned_paths.push_back(path);

        for (size_t i = 1; i < path.size(); ++i) {
            cout << "Now processing commodity " << commodity.source << " -> " << commodity.destination << " at path " << i << endl;
            auto e = boost::edge(path[i - 1], path[i], g).first;
            g[e].flow += commodity.demand;
			commodity.sent = g[e].flow;
        }
    }

    double prev_max_ratio = 0.0;
    while (true) {
        // Step 1: Calculate the bottleneck value
        double bottleneck_value = calculate_bottleneck(g);

        // Debugging: Print edge states
        //for (auto e : boost::make_iterator_range(boost::edges(g))) {
        //    auto source_node = boost::source(e, g);
        //    auto target_node = boost::target(e, g);

        //    // Get edge properties
        //    auto flow = g[e].flow;
        //    auto capacity = g[e].capacity;

        //    std::cout << source_node << " -> " << target_node
        //        << " [Flow: " << flow << ", Capacity: " << capacity << "]\n";
        //}

        // Step 2: Normalize flows using the bottleneck value
        if (isFlowExceedingCapacity(g)) {
            normalize_flows(g, bottleneck_value);
		    updateCommoditiesSent(commodities, bottleneck_value);
		}


        // Step 3: Recalculate weights
        recalculate_weights(g, alpha);

        // Step 4: Compute the maximum ratio after redistribution
        double max_ratio = 0.0;
        double highest_flow = 0.0;
        boost::graph_traits<Graph>::edge_iterator ei, ei_end;

        // Iterate through all edges in the graph
        for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei) {
            auto edge = g[*ei];  // Get the edge

            double total_flow_on_edge = edge.flow;  // Flow on this edge
            double edge_capacity = edge.capacity;  // Capacity of the edge

            // Check if this edge has more flow than the previous highest flow
            if (total_flow_on_edge > highest_flow) {
                highest_flow = total_flow_on_edge;

                // Calculate max ratio: capacity of bottleneck edge / total flow on the bottleneck edge
                max_ratio = edge_capacity / total_flow_on_edge;
            }
        }

        // Step 5: Check for convergence
        if (std::abs(max_ratio - prev_max_ratio) < epsilon) {
            solution = max_ratio;
            break;
        }
        prev_max_ratio = max_ratio;
    }

    return solution;
}