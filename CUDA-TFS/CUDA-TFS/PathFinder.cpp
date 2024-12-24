#include <vector>
#include <string>
#include <queue>
#include <limits>
#include <utility>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include "NetworkGraph.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>

const int INF = std::numeric_limits<int>::max();

using namespace std;

// Dijkstra's algorithm to find the shortest path
//std::vector<std::string> findShortestPath(const std::vector<Edge>& edges, const std::string& source, const std::string& destination) {
//    // Create adjacency list with weights
//    std::unordered_map<std::string, std::vector<std::pair<std::string, int>>> graph;
//    for (const auto& edge : edges) {
//        graph[edge.source].push_back({ edge.destination, edge.weight });
//    }
//
//    // Priority queue: (distance, node)
//    std::priority_queue<std::pair<int, std::string>, std::vector<std::pair<int, std::string>>, std::greater<>> pq;
//    std::unordered_map<std::string, int> distances;     // Distance from source to each node
//    std::unordered_map<std::string, std::string> prev; // To reconstruct the path
//
//    for (const auto& node : graph) {
//        distances[node.first] = INF;
//    }
//    distances[source] = 0;
//    pq.push({ 0, source });
//
//    while (!pq.empty()) {
//        auto [dist, current] = pq.top();
//        pq.pop();
//
//        // Skip processing if a shorter path is already found
//        if (dist > distances[current]) continue;
//
//        for (const auto& [neighbor, weight] : graph[current]) {
//            int newDist = dist + weight;
//            if (newDist < distances[neighbor]) {
//                distances[neighbor] = newDist;
//                prev[neighbor] = current;
//                pq.push({ newDist, neighbor });
//            }
//        }
//    }
//
//    std::vector<std::string> path;
//    for (std::string at = destination; at != ""; at = prev[at]) {
//        path.push_back(at);
//    }
//    std::reverse(path.begin(), path.end());
//
//    if (!path.empty() && path[0] == source) {
//        return path;
//    }
//    return {};
//}
//
//void findAllPathsHelper(const string& src, const string& dest,
//    const vector<Edge>& edges,
//    vector<string>& currentPath,
//    vector<vector<string>>& allPaths) {
//    currentPath.push_back(src);
//
//    if (src == dest) {
//        allPaths.push_back(currentPath);
//    }
//    else {
//        for (const auto& edge : edges) {
//            if (edge.source == src && find(currentPath.begin(), currentPath.end(), edge.destination) == currentPath.end()) {
//                findAllPathsHelper(edge.destination, dest, edges, currentPath, allPaths);
//            }
//        }
//    }
//
//    currentPath.pop_back();
//}
//
//vector<vector<string>> findAllPaths(const vector<Edge>& edges, const string& src, const string& dest) {
//    vector<vector<string>> allPaths;
//    vector<string> currentPath;
//
//    findAllPathsHelper(src, dest, edges, currentPath, allPaths);
//
//    // Sort paths by total weight (shortest to longest)
//    sort(allPaths.begin(), allPaths.end(), [&](const vector<string>& a, const vector<string>& b) {
//        int weightA = 0, weightB = 0;
//
//        for (size_t i = 0; i < a.size() - 1; ++i) {
//            for (const auto& edge : edges) {
//                if (edge.source == a[i] && edge.destination == a[i + 1]) {
//                    weightA += edge.weight;
//                }
//            }
//        }
//
//        for (size_t i = 0; i < b.size() - 1; ++i) {
//            for (const auto& edge : edges) {
//                if (edge.source == b[i] && edge.destination == b[i + 1]) {
//                    weightB += edge.weight;
//                }
//            }
//        }
//
//        return weightA < weightB; // Sort by ascending weight
//        });
//
//    return allPaths;
//}

std::vector<int> find_shortest_path(const Graph& g, int source, int destination) {
    // Prepare containers for the distances and predecessors
    std::vector<int> distance(boost::num_vertices(g), std::numeric_limits<int>::max()); // Use max() for infinity
    std::vector<int> predecessor(boost::num_vertices(g), -1);

    // Perform Dijkstra's algorithm
    boost::dijkstra_shortest_paths(g, source,
        boost::distance_map(&distance[0])
        .predecessor_map(&predecessor[0])
    );

    // Check if the destination is unreachable
    if (distance[destination] == std::numeric_limits<int>::max()) {
        std::cout << "No path exists from " << source << " to " << destination << ".\n";
        return {};
    }

    // Reconstruct the path from source to destination
    std::vector<int> path;
    for (int v = destination; v != -1; v = predecessor[v]) {
        path.push_back(v);
        // Break early if there's a loop or an invalid predecessor
        if (v == source) break;
    }

    std::reverse(path.begin(), path.end()); // Reverse the path to get the correct order
    return path;
}