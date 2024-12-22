#include <iostream>
#include "PathFinder.hpp"
#include "NetworkGraph.hpp"
#include "MPI_PropFlowAlgorithm.hpp"
#include <mpi.h>

int main(int argc, char* argv[]) {
	NetworkGraph graph;

    graph.addEdge("A", "B", 7, 16);
    graph.addEdge("A", "C", 4, 10);
    graph.addEdge("A", "D", 3, 9);
    graph.addEdge("B", "D", 4, 9);
    graph.addEdge("B", "E", 6, 13);
    graph.addEdge("C", "D", 2, 6);
    graph.addEdge("C", "F", 10, 20);
    graph.addEdge("D", "E", 7, 15);
    graph.addEdge("E", "G", 2, 6);
    graph.addEdge("E", "F", 4, 10);
    graph.addEdge("F", "G", 3, 8);
    //graph.addEdge("F", "I", 8, 20); 
    graph.addEdge("G", "B", 7, 18);
    /*graph.addEdge("G", "H", 5, 12);
    graph.addEdge("G", "I", 6, 14);
    graph.addEdge("H", "B", 11, 22);
    graph.addEdge("H", "I", 5, 8); */

    // define commodities
    vector<pair<string, string>> commodities = {
        {"A", "G"},{"C", "F"},{"B", "G"},
    };
    vector<double> demands = { 20, 15, 20 };

	MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	double start = MPI_Wtime();
	// Call MPI proportional flow algorithm

	MPI_equalDistributionAlgorithm(graph, commodities, demands, rank, size);

	double end = MPI_Wtime();

	MPI_Finalize();
    if (rank == 0) {
		cout << "MPI runtime: " << end - start << endl;
    }

	return 0;
}
