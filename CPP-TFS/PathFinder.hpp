#ifndef PATH_FINDER_HPP
#define PATH_FINDER_HPP

#include <vector>
#include "NetworkGraph.hpp"

using namespace std;

typedef graph_traits<Graph>::vertex_descriptor Vertex;
typedef std::pair<int, int> Edge;

std::vector<int> reconstruct_path(int source, int destination, const std::vector<Vertex>& predecessors);
std::vector<std::vector<std::vector<int>>> find_all_shortest_paths(const Graph& g);

#endif