// Commodity.cpp
#include "Commodity.hpp"
#include "NetworkGraph.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <iostream>
#include <ctime>

using namespace std;

vector<Commodity> generate_random_commodities(int num_commodities, const Graph& g, int min_units, int max_units) {
    vector<Commodity> commodities;

    // random number generator setup
    //srand(time(0));

    vector<int> valid_nodes;
    boost::graph_traits<Graph>::vertex_iterator vi, vi_end;
    boost::tie(vi, vi_end) = boost::vertices(g);

    for (auto v = vi; v != vi_end; ++v) {
        valid_nodes.push_back(*v);  
    }

    for (int i = 0; i < num_commodities; ++i) {
        // random selection of a source node from valid nodes
        int source = valid_nodes[rand() % valid_nodes.size()];

        // random assignment of destination node, as long as it is different from the source
        int destination = source;
        while (destination == source) {
            destination = valid_nodes[rand() % valid_nodes.size()];
        }

		// random number with range
        int demand = rand() % (max_units - min_units + 1) + min_units;

        commodities.push_back(Commodity(source, destination, demand));
    }

    return commodities;
}

void displayCommodityPaths(const Graph& g, const vector<Commodity>& commodities) {
	cout << "\n== Paths for Commodities after Flow Distribution ==" << endl;
    for (const auto& commodity : commodities) {
        cout << "Commodity from " << commodity.source
            << " to " << commodity.destination << ":\n";
        cout << "Initial Demand: " << commodity.init_demand
            << ", Total Sent: " << commodity.sent << "\n";

        for (const auto& path_with_flow : commodity.used_paths_with_flows) {
            const auto& path = path_with_flow.first;  // Path as a vector of nodes
            double flow_on_path = path_with_flow.second;

			if (flow_on_path == 0) continue;

            cout << "  Path: ";
            for (size_t i = 0; i < path.size(); ++i) {
                cout << path[i];
                if (i < path.size() - 1) cout << " -> ";
            }

            cout << "\n    Flow on Path: " << flow_on_path;

            // Display edge capacities and flows along this path
            cout << "\n    Edges on Path:\n";
            for (size_t i = 1; i < path.size(); ++i) {
                auto edge = boost::edge(path[i - 1], path[i], g).first;
                cout << "      " << path[i - 1] << " -> " << path[i]
                    << " [Flow: " << flow_on_path
                    << ", Capacity: " << g[edge].capacity << "]\n";
            }
            cout << endl;
        }

        cout << "=======================================================\n";
    }
}