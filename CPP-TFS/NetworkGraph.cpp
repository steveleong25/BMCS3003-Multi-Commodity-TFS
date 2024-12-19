#include "NetworkGraph.hpp"
#include <iostream>
#include <stdexcept>

Edge::Edge(std::string src, std::string dest, int wt, int cap)
    : source(std::move(src)), destination(std::move(dest)), weight(wt), capacity(cap), flow(0) {}

void NetworkGraph::addEdge(const std::string& src, const std::string& dest, int weight, int capacity) {
    //forward
    edges.emplace_back(src, dest, weight, capacity);

    //backward
    edges.emplace_back(dest, src, weight, capacity);
}


// Retrieve all edges in the graph
const std::vector<Edge> &NetworkGraph::getEdges() const {
    return edges;
}

// Retrieve an edge given its source and destination
Edge &NetworkGraph::getEdge(const std::string& src, const std::string& dest) {
    for (auto &edge : edges) {
        if (edge.source == src && edge.destination == dest) {
            return edge;
        }
    }
    throw std::runtime_error("Edge not found.");
}

// Display the graph (for debugging purposes)
void NetworkGraph::displayGraph() const {
    std::cout << "Graph Edges:\n";
    for (const auto& edge : edges) {
        std::cout << edge.source << " -> " << edge.destination
            << " | Weight: " << edge.weight
            << ", Capacity: " << edge.capacity
            << ", Flow: " << edge.flow << '\n';
    }
}
