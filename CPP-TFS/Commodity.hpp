// Commodity.hpp
#ifndef COMMODITY_HPP
#define COMMODITY_HPP

#include "NetworkGraph.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <vector>

struct Commodity {
    int source;
    int destination;
    int demand;

    Commodity(int src, int dest, int demand_val)
        : source(src), destination(dest), demand(demand_val) {
    }
};

bool has_edge_between(int source, int destination, const Graph& g);

// generate random commodities
std::vector<Commodity> generate_random_commodities(int num_commodities, const Graph& g);

#endif

