#include "NetworkGraph.hpp"
#include <iostream>


Edge::Edge(std::string src, std::string dest, int wt, int cap)
    : source(std::move(src)), destination(std::move(dest)), weight(wt), capacity(cap), flow(0) {}

void NetworkGraph::addEdge(const std::string& src, const std::string& dest, int weight, int capacity) {
    //forward
    edges.emplace_back(src, dest, weight, capacity);

    //backward
    edges.emplace_back(dest, src, weight, capacity);
}


// get all edges
std::vector<Edge> &NetworkGraph::getEdges() {
    return edges;
}

// get edge by source and dest
Edge &NetworkGraph::getEdge(const std::string& src, const std::string& dest) {
    for (auto &edge : edges) {
        if (edge.source == src && edge.destination == dest) {
            return edge;
        }
    }
    throw std::runtime_error("Edge not found.");
}

// reset flow of all edges
void NetworkGraph::resetFlow() {
	for (auto& edge : edges) {
		edge.flow = 0;
	}
}