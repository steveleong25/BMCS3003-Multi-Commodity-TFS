#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <unordered_map>
#include <algorithm>
#include "NetworkGraph.hpp"
#include "PathFinder.hpp"

using namespace std;

const double INF = numeric_limits<double>::max();


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
void equalDistributionAlgorithm(NetworkGraph& graph,
    std::vector<std::pair<std::string, std::string>> commodities,
    std::vector<double> demands) {
    std::vector<double> flowDelivered(commodities.size(), 0.0); // Track flow delivered for each commodity

    for (size_t i = 0; i < commodities.size(); ++i) {
        const std::string& source = commodities[i].first;
        const std::string& destination = commodities[i].second;

        double remainingDemand = demands[i];

        // Get all possible paths from source to destination, ranked by weight (shortest to longest)
        std::vector<std::vector<std::string>> allPaths = findAllPaths(graph.getEdges(), source, destination);

        for (const auto& path : allPaths) {
            if (remainingDemand <= 0) break;

            // Calculate bottleneck capacity for the current path
            double pathCapacity = INF;
            for (size_t j = 0; j < path.size() - 1; ++j) {
                Edge& edge = graph.getEdge(path[j], path[j + 1]);
                pathCapacity = std::min(pathCapacity, static_cast<double>(edge.capacity - edge.flow));
            }

            if (pathCapacity > 0) {
                // Assign flow to the path
                double flowToSend = std::min(remainingDemand, pathCapacity);
                sendFlow(graph, path, flowToSend);

                // Update remaining demand and track delivered flow
                remainingDemand -= flowToSend;
                flowDelivered[i] += flowToSend;
            }
        }
    }

    // Display results
    std::cout << "\nFinal Results: Units Successfully Reaching Destinations\n";
    for (size_t i = 0; i < commodities.size(); ++i) {
        std::cout << "Commodity " << i + 1 << " (From " << commodities[i].first
            << " to " << commodities[i].second << "): "
            << flowDelivered[i] << " units\n";
    }
}
