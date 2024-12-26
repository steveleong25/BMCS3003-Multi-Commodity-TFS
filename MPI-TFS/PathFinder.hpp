#ifndef PATH_FINDER_HPP
#define PATH_FINDER_HPP

#include <vector>
#include "NetworkGraph.hpp"

using namespace std;

vector<int> find_shortest_path(const Graph& g, int source, int destination);

#endif