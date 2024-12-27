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
    const double init_demand;
    double demand;
    double sent;

    Commodity(int src, int dest, int demand_val)
        : source(src), destination(dest), demand(demand_val), sent(0), init_demand(demand_val) {
    }
};

// generate random commodities
std::vector<Commodity> generate_random_commodities(int num_commodities, const Graph& g, int min_units, int max_units);

#endif

