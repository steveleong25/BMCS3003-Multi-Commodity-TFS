#include "NetworkGraph.hpp"

// Add edge
void NetworkGraph::addEdge(const std::string& src, const std::string& dest, int distance, int capacity, bool isBidirectional) {
    // One-way edges
    edges.push_back(Edge(src, dest, distance, capacity));

    // Two-way edges
    if (isBidirectional) {
        edges.push_back(Edge(dest, src, distance, capacity));
    }
}

void NetworkGraph::displayGraph() const {
    std::cout << "Graph Edges:" << std::endl;
    for (const auto& edge : edges) {
        std::cout << edge.source << " -> " << edge.destination
                  << " [Distance: " << edge.distance
                  << ", Capacity: " << edge.capacity << "]" << std::endl;
    }
}
