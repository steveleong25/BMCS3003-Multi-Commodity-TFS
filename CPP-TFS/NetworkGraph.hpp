#ifndef NETWORK_GRAPH_HPP
#define NETWORK_GRAPH_HPP

#include <string>
#include <vector>

// Define the Edge structure
struct Edge {
    std::string source;
    std::string destination;
    int weight;
    int capacity;
    int flow;

    Edge(std::string src, std::string dest, int wt, int cap);

    void setFlow(int newFlow) {
        flow = newFlow;
    }
};

// Define the NetworkGraph class
class NetworkGraph {
private:
    std::vector<Edge> edges;  // Edge list

public:
    void addEdge(const std::string &src, const std::string &dest, int weight, int capacity, bool isBidirectional = false);
    const std::vector<Edge> &getEdges() const;
    Edge &getEdge(const std::string &src, const std::string &dest);
    void displayGraph() const;
};

#endif
