#include <vector>
#include <iostream>
#include "NetworkGraph.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>

const int INF = std::numeric_limits<int>::max();

using namespace std;

typedef graph_traits<Graph>::vertex_descriptor Vertex;
typedef std::pair<int, int> Edge;

// Function to reconstruct a single path using the predecessor map
std::vector<int> reconstruct_path(int source, int destination, const std::vector<Vertex>& predecessors) {
    std::vector<int> path;
    for (Vertex v = destination; v != source; v = predecessors[v]) {
        path.push_back(v);
    }
    path.push_back(source);
    std::reverse(path.begin(), path.end()); // Reverse to get the path from source to destination
    return path;
}

// Function to compute and return all shortest paths
std::vector<std::vector<std::vector<int>>> find_all_shortest_paths(const Graph& g) {
    size_t num_vertices = boost::num_vertices(g);

    // Nested vector to store all paths
    // Outer vector: Source vertices
    // Inner vector: Target vertices
    // Innermost vector: Shortest path from source to target
    std::vector<std::vector<std::vector<int>>> all_paths(num_vertices, std::vector<std::vector<int>>(num_vertices));

    for (Vertex source = 0; source < num_vertices; ++source) {
        std::vector<int> distances(num_vertices, INF);
        std::vector<Vertex> predecessors(num_vertices, source);

        // Run Dijkstra's algorithm from the current source vertex
        boost::dijkstra_shortest_paths(g, source,
            boost::distance_map(boost::make_iterator_property_map(distances.begin(), boost::get(boost::vertex_index, g))).
            predecessor_map(boost::make_iterator_property_map(predecessors.begin(), boost::get(boost::vertex_index, g))).
            weight_map(boost::get(&EdgeProperties::weight, g)));

        // Reconstruct paths to all other vertices
        for (Vertex target = 0; target < num_vertices; ++target) {
            if (source == target) {
                all_paths[source][target] = { static_cast<int>(source) }; // Path to itself
            }
            else if (distances[target] == INF) {
                all_paths[source][target] = {}; // No path exists
            }
            else {
                all_paths[source][target] = reconstruct_path(source, target, predecessors);
            }
        }
    }

    return all_paths;
}