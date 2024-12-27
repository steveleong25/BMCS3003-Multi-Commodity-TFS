#include <mpi.h>
#include "NetworkGraph.hpp"
#include "PathFinder.hpp"
#include "MPI_FlowAlgorithm.hpp"
#include "Commodity.hpp"
#include <iostream>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/random.hpp>
#include <boost/random.hpp>

using namespace std;

using namespace boost;

Graph generate_random_graph(long long num_nodes, long long num_edges) {
    if (num_nodes <= 1) {
        throw std::invalid_argument("Number of nodes must be greater than 1.");
    }

    long long max_edges = num_nodes * (num_nodes - 1); // directed graph
    if (num_edges > max_edges) {
        std::cerr << "Requested number of edges (" << num_edges
            << ") exceeds the maximum possible edges (" << max_edges
            << ") for " << num_nodes << " nodes.\n";
        num_edges = max_edges; // adjust num_edges to the maximum
        std::cout << "Reducing number of edges to " << num_edges << ".\n";
    }

    Graph g(num_nodes);
    boost::random::mt19937 gen;
    boost::random::uniform_int_distribution<> dist_weight(1, 20);  
    boost::random::uniform_int_distribution<> dist_capacity(10, 50); 

    long long edge_count = 0;

    cout << "Initializing edges..." << endl;
    // generate random edges
    while (edge_count < num_edges) {
        int u = gen() % num_nodes; //source
        int v = gen() % num_nodes; //destination

        if (u != v) {
            auto [edge, exists] = boost::edge(u, v, g);
            if (!exists) {
                // Add the edge if it doesn't exist
                auto e = boost::add_edge(u, v, g).first;
                g[e].capacity = dist_capacity(gen);  // Set the capacity for the edge
                g[e].flow = 0;  // Initialize the flow to 0

                put(boost::edge_weight, g, e, dist_weight(gen));
                edge_count++;
            }
        }
    }

    cout << "Graph with " << num_nodes << " nodes and " << num_edges << " edges created." << endl;

    return g;
}

Graph graph_test_init() {
    std::vector<std::pair<int, int>> edges = {
        {0, 1}, {1, 0}, // 0 and 1
        {1, 2}, {2, 1}, // 1 and 2
        {2, 3}, {3, 2}, // 2 and 3
        {2, 5}, {5, 2}, // 2 and 5
        {4, 1}, {1, 4}  // 4 and 1
    };

    std::vector<EdgeProperties> edge_properties = {
        {10, 10, 0}, {10, 10, 0}, //  0 <-> 1
        {12, 20, 0}, {12, 20, 0}, //  1 <-> 2
        {8, 15, 0}, {8, 15, 0},   //  2 <-> 3
        {10, 40, 0}, {10, 40, 0}, //  2 <-> 5
        {15, 30, 0}, {15, 30, 0}  //  4 <-> 1
    };

    Graph g;

    for (size_t i = 0; i < edges.size(); ++i) {
        auto e = boost::add_edge(edges[i].first, edges[i].second, g).first;
        g[e].capacity = edge_properties[i].capacity;
        g[e].flow = edge_properties[i].flow;
        put(boost::edge_weight, g, e, edge_properties[i].weight);
    }

    return g;
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Graph g;
    vector<Commodity> commodities;
    double epsilon = 0.01;
    double alpha = 0.1;

    // Initialize graph and commodities (only on rank 0)
    if (rank == 0) {
        try {
            //Graph
            long long num_nodes = 10000; // adjust nodes
            long long num_edges = 4000000; // desired number of edges
            g = graph_test_init();    
            //g = generate_random_graph(num_nodes, num_edges);
            Graph g2 = g;

            //Commodity
            //int num_commodities = 6;  // number of commodities
            //int min_demand = 10;      // minimum demand for a commodity
            //int max_demand = 100;     // maximum demand for a commodity

            graph_traits<Graph>::vertex_iterator vi, vi_end;
            tie(vi, vi_end) = boost::vertices(g);

            //std::vector<Commodity> commodities = generate_random_commodities(num_commodities, g, min_demand, max_demand);
            std::vector<Commodity> commodities = {
                {0, 3, 20},
                {4, 5, 5},
            };
        }
        catch (const std::invalid_argument& e) {
            std::cerr << "Error: " << e.what() << endl;
        }
    }

    // Call parallelized algorithm
    double solution = MPI_flowDistributionAlgorithm(g, commodities, epsilon, alpha, rank, size);

    if (rank == 0) {
        cout << "Solution: " << solution << endl;
    }

    MPI_Finalize();
    return 0;
	
	return 0;
}
