#ifndef NETWORK_GRAPH_HPP
#define NETWORK_GRAPH_HPP

#include <string>
#include <vector>
#include <boost/graph/adjacency_list.hpp>

using namespace boost;

struct EdgeProperties {
    double weight;
    double capacity;
    double flow;

    EdgeProperties() : weight(0), capacity(0), flow(0) {}
	EdgeProperties(double cap, double f = 0) : weight(1/cap), capacity(cap), flow(f) {}
};

// define the graph type (using adjacency list)
typedef boost::adjacency_list<
    boost::vecS,             // storage for vertices (using vector)
    boost::vecS,             // storage for edges (using vector)
    boost::directedS,        // directed graph
    boost::no_property,      // vertex properties (no extra properties here)
    boost::property<boost::edge_weight_t, int, EdgeProperties>  // edge properties (weight and flow)
> Graph;

#endif
