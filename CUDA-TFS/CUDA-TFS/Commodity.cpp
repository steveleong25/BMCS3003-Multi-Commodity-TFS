// Commodity.cpp
#include "Commodity.hpp"
#include "NetworkGraph.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <cstdlib>
#include <ctime>

std::vector<Commodity> generate_random_commodities(int num_commodities, const Graph& g) {
    std::vector<Commodity> commodities;

    // Random number generator setup
    //std::srand(std::time(0));

    std::vector<int> valid_nodes;
    boost::graph_traits<Graph>::vertex_iterator vi, vi_end;
    boost::tie(vi, vi_end) = boost::vertices(g);

    for (auto v = vi; v != vi_end; ++v) {
        valid_nodes.push_back(*v);  
    }

    for (int i = 0; i < num_commodities; ++i) {
        // random selection of a source node from valid nodes
        int source = valid_nodes[std::rand() % valid_nodes.size()];

        // random assignment of destination node, as long as it is different from the source
        int destination = source;
        while (destination == source) {
            destination = valid_nodes[std::rand() % valid_nodes.size()];
        }

		// range of 10 to 50
        int demand = std::rand() % 40 + 10;

        commodities.push_back(Commodity(source, destination, demand));
    }

    return commodities;
}