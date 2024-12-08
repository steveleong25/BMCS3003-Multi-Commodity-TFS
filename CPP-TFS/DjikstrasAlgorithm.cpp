#include <vector>
#include <string>
#include <unordered_map>
#include <queue>
#include <limits>
#include <utility>
#include <algorithm>
#include <iostream>
#include "NetworkGraph.hpp"

const int INF = std::numeric_limits<int>::max();

// Dijkstra's algorithm to find the shortest path
std::vector<std::string> findShortestPath(const std::vector<Edge>& edges, const std::string& source, const std::string& destination) {
    // Create adjacency list with weights
    std::unordered_map<std::string, std::vector<std::pair<std::string, int>>> graph;
    for (const auto& edge : edges) {
        graph[edge.source].push_back({ edge.destination, edge.weight });
    }

    // Priority queue: (distance, node)
    std::priority_queue<std::pair<int, std::string>, std::vector<std::pair<int, std::string>>, std::greater<>> pq;
    std::unordered_map<std::string, int> distances;     // Distance from source to each node
    std::unordered_map<std::string, std::string> prev; // To reconstruct the path

    for (const auto& node : graph) {
        distances[node.first] = INF;
    }
    distances[source] = 0;
    pq.push({ 0, source });

    while (!pq.empty()) {
        auto [dist, current] = pq.top();
        pq.pop();

        // Skip processing if a shorter path is already found
        if (dist > distances[current]) continue;

        for (const auto& [neighbor, weight] : graph[current]) {
            int newDist = dist + weight;
            if (newDist < distances[neighbor]) {
                distances[neighbor] = newDist;
                prev[neighbor] = current;
                pq.push({ newDist, neighbor });
            }
        }
    }

    std::vector<std::string> path;
    for (std::string at = destination; at != ""; at = prev[at]) {
        path.push_back(at);
    }
    std::reverse(path.begin(), path.end());

    if (!path.empty() && path[0] == source) {
        return path; 
    }
    return {};
}
