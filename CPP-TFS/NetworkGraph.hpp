#ifndef NETWORK_GRAPH_HPP
#define NETWORK_GRAPH_HPP

#include <string>
#include <vector>
#include <iostream>

// Define the Edge structure
struct Edge {
    std::string source;
    std::string destination;
    int distance;    
    int capacity;  

    Edge(std::string src, std::string dest, int dt, int cap)
        : source(src), destination(dest), distance(dt), capacity(cap) {}
};

// Define the NetworkGraph class
class NetworkGraph {
private:
    std::vector<Edge> edges; 

public:
    void addEdge(const std::string& src, const std::string& dest, int distance, int capacity, bool isBidirectional = false);
    void displayGraph() const;
    std::vector<Edge> getEdges() const { return edges; }
};

#endif