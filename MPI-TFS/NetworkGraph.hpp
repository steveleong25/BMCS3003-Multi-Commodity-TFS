#ifndef NETWORK_GRAPH_HPP
#define NETWORK_GRAPH_HPP

#include <string>
#include <vector>
#include <boost/graph/adjacency_list.hpp>

using namespace boost;

struct EdgeProperties {
    int weight;
    int capacity;
    int flow;

    EdgeProperties() : weight(0), capacity(0), flow(0) {}
    EdgeProperties(int w, int cap, int f = 0) : weight(w), capacity(cap), flow(f) {}
};

// Define the graph type (using adjacency list)
typedef boost::adjacency_list<
    boost::vecS,             // Storage for vertices (using vector)
    boost::vecS,             // Storage for edges (using vector)
    boost::directedS,        // Directed graph
    boost::no_property,      // Vertex properties (no extra properties here)
    boost::property<boost::edge_weight_t, int, EdgeProperties>  // Edge properties (weight and flow)
> Graph;

#endif
