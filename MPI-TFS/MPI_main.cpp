#include <mpi.h>
#include "NetworkGraph.hpp"
#include "MPI_FlowAlgorithm.hpp"
#include "Commodity.hpp"
#include <iostream>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/random.hpp>
#include <boost/random.hpp>
#include "PathFinder.hpp"

using namespace std;
using namespace boost;

Graph generate_random_graph(long long num_nodes, long long num_edges) {
    Graph g(num_nodes);
    boost::random::mt19937 gen;
    boost::random::uniform_int_distribution<> dist_weight(1, 20);
    boost::random::uniform_int_distribution<> dist_capacity(10, 50);

    long long edge_count = 0;
    while (edge_count < num_edges) {
        int u = gen() % num_nodes;
        int v = gen() % num_nodes;
        if (u != v) {
            auto [edge, exists] = boost::edge(u, v, g);
            if (!exists) {
                auto e = boost::add_edge(u, v, g).first;
                g[e].capacity = dist_capacity(gen);
                g[e].flow = 0;
                put(boost::edge_weight, g, e, dist_weight(gen));
                edge_count++;
            }
        }
    }
    return g;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Graph g;
    std::vector<Commodity> commodities = { {0, 3, 20}, {4, 5, 10} };
    double epsilon = 0.01;
    double alpha = 0.1;

    if (rank == 0) {
        g = generate_random_graph(100, 500);
        //commodities = { {0, 3, 20}, {4, 5, 10} };
    }

    MPI_Bcast(&g, sizeof(Graph), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&commodities, commodities.size() * sizeof(Commodity), MPI_BYTE, 0, MPI_COMM_WORLD);

    double solution = MPI_flowDistributionAlgorithm(g, commodities, epsilon, alpha);

    if (rank == 0) {
        cout << "Final Solution (Max Ratio): " << solution << endl;
    }

    MPI_Finalize();
    return 0;
}
