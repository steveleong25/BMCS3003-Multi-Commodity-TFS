#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <unordered_map>
#include <algorithm>
#include "NetworkGraph.hpp"
#include "DjikstrasAlgorithm.hpp"


using namespace std;

const double INF = numeric_limits<double>::max();

// Max congestion finder
double findMaxCongestion(NetworkGraph& graph) {
    double maxCongestion = 0;
    for (const Edge& e : graph.getEdges()) {
        if (e.capacity > 0) {
            maxCongestion = max(maxCongestion, static_cast<double>(e.flow) / e.capacity);
        }
    }
    return maxCongestion;
}

// Scale down flows
void scaleDownFlows(NetworkGraph& graph, double scaleFactor) {
    for (const Edge& e : graph.getEdges()) {  // Read-only access to edges
        if (e.capacity > 0) {
            Edge& edge = graph.getEdge(e.source, e.destination);
            edge.setFlow(static_cast<int>(e.flow / scaleFactor));
        }
    }
}

// Send flow along a path
void sendFlow(NetworkGraph& graph, const vector<string>& path, double amount) {
    for (size_t i = 0; i < path.size() - 1; ++i) {
        Edge& edge = graph.getEdge(path[i], path[i + 1]);
        edge.flow += amount;
    }
}

// Equal Distribution Algorithm
void equalDistributionAlgorithm(NetworkGraph& graph, vector<pair<string, string>> commodities, vector<double> demands) {
    int maxIterations = 100;

    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        for (size_t i = 0; i < commodities.size(); ++i) {
            const string& source = commodities[i].first;
            const string& target = commodities[i].second;

            vector<string> path = findShortestPath(graph.getEdges(), source, target);

            if (!path.empty()) {
                double remainingDemand = demands[i];
                double pathCapacity = INF;

                for (size_t j = 0; j < path.size() - 1; ++j) {
                    Edge& edge = graph.getEdge(path[j], path[j + 1]);
                    pathCapacity = min(pathCapacity, static_cast<double>(edge.capacity - edge.flow));
                }

                double flowToSend = min(remainingDemand, pathCapacity);
                sendFlow(graph, path, flowToSend);
                demands[i] -= flowToSend;
            }
        }

        double maxCongestion = findMaxCongestion(graph);

        if (maxCongestion <= 1) {
            break;
        }

        scaleDownFlows(graph, maxCongestion);
    }
}

