#ifndef DJIKSTRAS_ALGORITHM_HPP
#define DJIKSTRAS_ALGORITHM_HPP

#include <vector>
#include <string>
#include "NetworkGraph.hpp"

// Declaration of the findShortestPath function
std::vector<std::string> findShortestPath(const std::vector<Edge>& edges, const std::string& source, const std::string& destination);

#endif