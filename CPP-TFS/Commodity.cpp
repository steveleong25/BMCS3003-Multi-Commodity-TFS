// Commodity.cpp
#include "Commodity.hpp"
#include "NetworkGraph.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <cstdlib>
#include <ctime>

bool has_edge_between(int source, int destination, const Graph& g) {
    typedef boost::graph_traits<Graph>::edge_descriptor Edge;
    Edge e;
    bool exists = boost::edge(source, destination, g).second;  // checks if an edge exists from source to destination
    return exists;
}

std::vector<Commodity> generate_random_commodities(int num_commodities, const Graph& g) {
    std::vector<Commodity> commodities;

    // Random number generator setup
    //std::srand(std::time(0));  // seed the random number generator

    // Create a vector to store valid nodes from the graph
    std::vector<int> valid_nodes;
    boost::graph_traits<Graph>::vertex_iterator vi, vi_end;
    boost::tie(vi, vi_end) = boost::vertices(g);

    // Iterate through all the vertices in the graph using the vertex iterator
    for (auto v = vi; v != vi_end; ++v) {
        valid_nodes.push_back(*v);  // Add vertex to valid_nodes list
    }

    // Generate random commodities
    for (int i = 0; i < num_commodities; ++i) {
        // Randomly pick a source node from valid nodes
        int source = valid_nodes[std::rand() % valid_nodes.size()];

        // Randomly pick a destination node, ensuring it is different from the source
        int destination = source;
        while (destination == source) {
            destination = valid_nodes[std::rand() % valid_nodes.size()];
        }

        // Random demand in the given range
        int demand = std::rand() % 40 + 10;

        commodities.push_back(Commodity(source, destination, demand));
    }

    return commodities;
}