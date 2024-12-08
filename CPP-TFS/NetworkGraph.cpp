#include "NetworkGraph.hpp"

// Add edge
void NetworkGraph::addEdge(const std::string& src, const std::string& dest, int weight, int capacity, bool isBidirectional) {
    // One-way edges
    edges.push_back(Edge(src, dest, weight, capacity));

    // Two-way edges
    if (isBidirectional) {
        edges.push_back(Edge(dest, src, weight, capacity));
    }
}

void NetworkGraph::displayGraph() const {
    std::cout << "Graph Edges:" << std::endl;
    for (const auto& edge : edges) {
        std::cout << edge.source << " -> " << edge.destination
                  << " [Distance: " << edge.weight
                  << ", Capacity: " << edge.capacity << "]" << std::endl;
    }
}

