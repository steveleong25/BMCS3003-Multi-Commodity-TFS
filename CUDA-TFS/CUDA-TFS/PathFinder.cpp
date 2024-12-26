#include <vector>
#include <iostream>
#include "NetworkGraph.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>

const int INF = std::numeric_limits<int>::max();

using namespace std;

std::vector<int> find_shortest_path(const Graph& g, int source, int destination) {
    std::vector<int> distance(boost::num_vertices(g), std::numeric_limits<int>::max()); 
    std::vector<int> predecessor(boost::num_vertices(g), -1);

    // Dijkstra's algorithm
    boost::dijkstra_shortest_paths(g, source,
        boost::distance_map(&distance[0])
        .predecessor_map(&predecessor[0])
    );

    // if the destination is unreachable
    if (distance[destination] == std::numeric_limits<int>::max()) {
        std::cout << "No path exists from " << source << " to " << destination << ".\n";
        return {};
    }

    // reconstruct path from source to destination
    std::vector<int> path;
    for (int v = destination; v != -1; v = predecessor[v]) {
        path.push_back(v);
        // invalid predecessor then break
        if (v == source) break;
    }

    std::reverse(path.begin(), path.end()); // reverse path
    return path;
}