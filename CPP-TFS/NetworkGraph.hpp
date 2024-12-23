#ifndef NETWORK_GRAPH_HPP
#define NETWORK_GRAPH_HPP

#include <string>
#include <vector>
#include <boost/graph/adjacency_list.hpp>

using namespace boost;

//struct Edge {
//    std::string source;
//    std::string destination;
//    int weight;    // used for shortest path calculations
//    int capacity;  // max flow capacity of the edge
//    int flow;      // current flow of the edge
//
//    Edge(std::string src, std::string dest, int wt, int cap);
//
//    void setFlow(int newFlow) {
//        flow = newFlow;
//    }
//};
//
//class NetworkGraph {
//private:
//    std::vector<Edge> edges;
//
//public:
//    void addEdge(const std::string &src, const std::string &dest, int weight, int capacity);
//    std::vector<Edge> &getEdges();
//    Edge &getEdge(const std::string &src, const std::string &dest);
//    void resetFlow();
//};

struct EdgeProperties {
    int weight;
    int capacity;
    int flow;  // The current flow on the edge

    EdgeProperties() : weight(0), capacity(0), flow(0) {}
	EdgeProperties(int w, int cap, int f = 0) : weight(w), capacity(cap), flow(f) {}
};

// Define the graph type (using adjacency list)
typedef boost::adjacency_list<
    boost::vecS,             // Storage for vertices (using vector)
    boost::vecS,             // Storage for edges (using vector)
    boost::directedS,        // Bidirectional graph
    boost::no_property,      // Vertex properties (no extra properties here)
    boost::property<boost::edge_weight_t, int, EdgeProperties>  // Edge properties (weight and flow)
> Graph;

#endif
