#include <vector>
#include <iostream>
#include "NetworkGraph.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>

const int INF = std::numeric_limits<int>::max();

using namespace std;


std::vector<int> find_shortest_path(const Graph& g, int source, int destination) {
    if (source < 0 || source >= boost::num_vertices(g) ||
        destination < 0 || destination >= boost::num_vertices(g)) {
        std::cerr << "Invalid source or destination vertex." << std::endl;
        return {}; // Return empty vector for invalid input
    }

    std::vector<int> distances(boost::num_vertices(g));
    std::vector<boost::graph_traits<Graph>::vertex_descriptor> predecessors(boost::num_vertices(g));

    boost::dijkstra_shortest_paths(g, source,
        boost::distance_map(boost::make_iterator_property_map(distances.begin(), boost::get(boost::vertex_index, g))).
        predecessor_map(boost::make_iterator_property_map(predecessors.begin(), boost::get(boost::vertex_index, g))).
        weight_map(boost::get(&EdgeProperties::weight, g)));

    if (distances[destination] == std::numeric_limits<int>::max()) {
        std::cout << "No path exists.\n";
        return {}; // Return empty vector if no path exists
    }

    std::vector<int> path;
    boost::graph_traits<Graph>::vertex_descriptor current = destination;
    while (current != source) {
        path.push_back(current);
        current = predecessors[current];
    }
    path.push_back(source);
    std::reverse(path.begin(), path.end());

    return path;
}