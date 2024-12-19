#ifndef NETWORK_GRAPH_HPP
#define NETWORK_GRAPH_HPP

#include <string>
#include <vector>

struct Edge {
    std::string source;
    std::string destination;
    int weight;    // used for shortest path calculations
    int capacity;  // max flow capacity of the edge
    int flow;      // current flow of the edge

    Edge(std::string src, std::string dest, int wt, int cap);

    void setFlow(int newFlow) {
        flow = newFlow;
    }
};

class NetworkGraph {
private:
    std::vector<Edge> edges;  // Edge list

public:
    void addEdge(const std::string &src, const std::string &dest, int weight, int capacity);
    const std::vector<Edge> &getEdges() const;
    Edge &getEdge(const std::string &src, const std::string &dest);
    void displayGraph() const;
};

#endif
