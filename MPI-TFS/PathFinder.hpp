#ifndef PATH_FINDER_HPP
#define PATH_FINDER_HPP

#include <vector>
#include <string>
#include <unordered_set>
#include "NetworkGraph.hpp"

using namespace std;

// Declaration of the findShortestPath function
vector<string> findShortestPath(const vector<Edge>& edges, const string& source, const string& destination);
void findAllPathsHelper(const string& src, const string& dest, const vector<Edge>& edges, vector<string>& currentPath, vector<vector<string>>& allPaths);
vector<vector<string>> findAllPaths(const vector<Edge>& edges, const string& src, const string& dest);

#endif